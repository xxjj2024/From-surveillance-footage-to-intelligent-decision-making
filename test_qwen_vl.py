# -*- coding: utf-8 -*-
"""
============================================================================
 废旧机动车拆解零件拖带检测 —— Qwen2.5-VL (Ollama) 批量图片判定客户端
============================================================================
 功能：
  - 从本地图片文件夹读取图片，逐张发送到 Ollama 服务端
    Ollama 部署的 qwen2.5vl:7b 视觉语言模型；
  - 用反复打磨的判定提示词（边界清晰）判断：
    画面中的机动车是否拖带/悬挂/装载废旧机动车拆解后的零碎零件；
  - 模型严格输出 JSON：{"answer": "是"/"否", "description": "..."}；
  - 全部结果汇总写入 CSV，方便事后筛选"是"的样本。

 前置条件（服务器端，SSH 上去执行一次）：
  1. Ollama 已安装且 qwen2.5vl:7b 已拉取：  ollama list
  2. Ollama 监听所有网卡（否则本地无法远程连接）：
     sudo systemctl edit ollama
     写入：
       [Service]
       Environment="OLLAMA_HOST=0.0.0.0"
     然后  sudo systemctl restart ollama
  3. 防火墙放行 11434 端口：
     sudo ufw allow 11434/tcp

 使用方法：
  python test_qwen_vl.py               # 扫描全部裁剪图
  python test_qwen_vl.py 50            # 只扫前 50 张（快速试跑）
============================================================================
"""

import base64
import csv
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime

# ======================= 配置区域 =======================
# ★ 发布前请替换以下内容为你自己的环境信息 ★
#
# Ollama 服务地址，两种典型配置：
#   A. Ollama 与本脚本同机：      http://127.0.0.1:11434/api/chat
#   B. Ollama 在远程服务器：
#      - 直连：http://服务器IP:11434/api/chat（需监听0.0.0.0且放行防火墙）
#      - 隧道（更安全）：ssh -N -L 本地端口:127.0.0.1:11434 用户名@服务器IP
#        保持SSH存活后，OLLAMA_URL 填 http://127.0.0.1:本地端口/api/chat
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"   # ← 按上面说明替换为你的地址

# 模型名（ollama list 里显示的名称）
MODEL = "qwen2.5vl:latest"

# 待判定图片文件夹
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "truck_crops")   # ← 替换为你的图片文件夹

# 结果 CSV 保存路径（带版本号，v4 提示词结果）
RESULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "判定结果.csv")

# 网络超时（秒）：VLM 推理较慢，7B 模型单张约 10~40 秒
REQUEST_TIMEOUT = 120

# 失败重试次数
MAX_RETRIES = 3
# ===========================================================

# ---------------------------------------------------------------------------
# 判定提示词（v4，在v3基础上强化轮胎堆辨别）：
# 人工反馈的五类误报及对应修正：
#   误报1：拉的锅碗瓢盆（生活废品）→ 新增排除：生活日用品/厨具/容器
#   误报2：拉完整汽车             → 强化排除：完整整车（已有但模型仍误判）
#   误报3：拉白色袋装物           → 强化排除：袋装/箱装/封闭容器内货物
#   误报4：拉纸箱子               → 强化排除：纸箱明确点名
#   误报5：拉的轮胎/不是拉东西的   → 排除纯轮胎堆；要求必须有可见拖带/装载行为
# 人工确认的正样本特征（作为"是"的锚定描述）：
#   拖车/车斗内堆积大量形状不规则、颜色偏暗的零碎金属部件
#   （车轮部件、发动机残骸、车架碎片等拆解产物）。
#   注意：零件"整体装在车斗/拖车里运输"也算"是"（不要求散落地面被单独拖拽）。
# ---------------------------------------------------------------------------
PROMPT = """你是一名废旧机动车回收拆解场的监控审核员。请仔细观察这张车辆照片。

判定目标：这辆机动车是否正在【拖带、悬挂、绑扎、装载或运输废旧机动车被拆解后产生的零碎金属零件】。

★ 第一步：确认车辆确实在"拉东西"
画面中必须能看到明确的拖带/装载行为：车斗或货箱内有堆积物、车后有被拖挂的载物拖车/小推车、或车尾绑扎悬挂着物品。如果车辆空载、无法看清车斗内部、或看不出在拉任何东西，直接判"否"。

★ 第二步：仔细辨认装载物的【材质与颜色】，按下面标准判定

判定为【是】的唯一标准：
被拖带/装载的是【大量形状不规则、颜色偏暗的"金属"部件——必须呈现金属质感（深灰/锈红/黑色/暗银色，有反光或锈迹）】，
典型如：车轮轮毂、发动机残骸、车架碎片、变速箱、钢梁铁块、成堆废旧金属件。
这些零件无论是散乱拖拽、悬挂绑扎，还是整体堆放在车斗/拖车里运输，都判"是"。

判定为【否】的情形（有其一即判否，必须严格排除）：
1. 拉的是【完整的整车】：被拖车辆有完整车身、车窗、轮胎齐整、外形是完整汽车轮廓，判否；
2. 拉的是【生活废品/日用品】：锅、碗、盆、桶、壶、家具、家电外壳等形状规整、颜色鲜艳的物品，判否；
3. 拉的是【农产品/食物/有机物】：蔬菜（白菜、大葱等绿色叶菜）、瓜果、粮食、袋装大米等
   ——绿色、白色、黄色等有机物颜色+不规则蓬松外形是蔬菜特征，绝对不是金属零件，判否；
4. 装载物被【袋装、箱装、苫布遮盖或封闭容器盛放】：内容物不可见或不是裸露金属零件，判否；
5. 拉的是【轮胎堆/大量轮胎】：黑色圆环状橡胶轮胎（无论数量多少、是否有轮毂）
   ——轮胎是橡胶材质，表面光滑无棱角、无金属锈迹，属于单一类型物品，
   即使颜色偏暗也绝对不是拆解金属零件，判否；
   注意区分：轮胎堆（全是圆环状物体）判否；轮胎"混在"大量不规则金属零件中且
   金属零件占主体，才可判是；
6. 拉的是【纯轮胎堆或单一类型非金属物品】：只有轮胎/木材/纸箱等单一物品，判否；
7. 画面模糊、距离太远、车斗内容物看不清时，一律判否。

★ 判定关键口诀：
"暗色不规则金属零件堆 = 是；完整整车/生活废品/蔬菜食物/轮胎堆/袋装箱装/看不清 = 否"
★ 材质辨别三要点：
- 金属零件：灰色系+硬朗棱角+反光锈迹，形状杂乱无规律；
- 蔬菜：绿色/白色叶片+蓬松柔软+无反光；
- 轮胎：黑色圆环状+橡胶质感+表面光滑+形状重复（一堆相似的圆环），
  轮胎堆的特征是"形状重复出现的圆环"，而金属零件堆是"形状各异的碎块"。

请严格按照以下 JSON 格式回答，不要输出任何其他内容：
{"answer": "是", "description": "用一句话描述画面中拆解金属零件的种类、颜色和所在位置"}
或
{"answer": "否", "description": "用一句话说明车辆在拉什么（完整整车/锅碗瓢盆/蔬菜/纸箱/袋装物/轮胎/空载等），以及为何不是拆解金属零件"}"""


def ask_vlm(image_path: str) -> dict:
    """
    调用远程 Ollama 的 qwen2.5vl 判定单张图片。
    返回 {"answer": "是"/"否"/"错误", "description": "..."}
    """
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
                "images": [img_b64],
            }
        ],
        "stream": False,
        # 低温度：判定类任务要稳定、少发挥
        "options": {"temperature": 0.1, "num_predict": 200},
    }).encode("utf-8")

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                OLLAMA_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = data.get("message", {}).get("content", "").strip()

            # 从回复中稳健地提取 JSON（模型偶尔会包一层 markdown 代码块）
            content = content.replace("```json", "").replace("```", "").strip()
            start, end = content.find("{"), content.rfind("}")
            if start != -1 and end != -1:
                parsed = json.loads(content[start:end + 1])
                answer = str(parsed.get("answer", "")).strip()
                desc = str(parsed.get("description", "")).strip()
                if answer in ("是", "否"):
                    return {"answer": answer, "description": desc}
            return {"answer": "解析失败", "description": content[:200]}

        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last_err = e
            print(f"    [重试 {attempt}/{MAX_RETRIES}] 请求失败: {e}", flush=True)
            time.sleep(3 * attempt)

    return {"answer": "网络错误", "description": str(last_err)}


def main() -> None:
    # 可选参数：只处理前 N 张
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if os.path.splitext(f)[1].lower() in exts
    )
    if limit:
        images = images[:limit]

    if not images:
        print(f"[!] {IMAGE_DIR} 下没有找到图片。")
        return

    print(f"[*] 待判定图片: {len(images)} 张")
    print(f"[*] 模型: {MODEL} @ {OLLAMA_URL}")
    print(f"[*] 结果将保存至: {RESULT_CSV}")
    print("[*] 提示：先跑少量试看效果，确认判定质量后再全量扫描。\n")

    # 先连通性自检：发一张图，若网络/模型不通直接终止
    print("[*] 连通性自检中...")
    test = ask_vlm(images[0])
    if test["answer"] == "网络错误":
        print(f"\n[!] 无法连接 Ollama，请检查：\n"
              f"    1. OLLAMA_URL 里的 IP 是否正确（当前 {OLLAMA_URL}）\n"
              f"    2. 服务器 Ollama 是否监听 0.0.0.0（见文件头部注释）\n"
              f"    3. 防火墙是否放行 Ollama 端口\n"
              f"    4. 服务器上  ollama list  是否有 {MODEL}")
        return
    print(f"[*] 自检通过。示例结果: {test['answer']} | {test['description'][:80]}\n")

    # ---------- 批量判定 ----------
    yes_count = 0
    with open(RESULT_CSV, "w", newline="", encoding="utf-8-sig") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["文件名", "判定", "模型描述", "耗时秒"])

        for i, img in enumerate(images, 1):
            t0 = time.time()
            result = ask_vlm(img)
            elapsed = time.time() - t0
            if result["answer"] == "是":
                yes_count += 1
            writer.writerow([os.path.basename(img), result["answer"],
                             result["description"], f"{elapsed:.1f}"])
            print(f"  ({i}/{len(images)}) {os.path.basename(img)} -> "
                  f"[{result['answer']}] {result['description'][:60]}  ({elapsed:.1f}s)",
                  flush=True)

    print(f"\n[完成] 共 {len(images)} 张，判定[是]: {yes_count} 张。")
    print(f"[完成] 结果已保存: {RESULT_CSV}（用 Excel 打开筛'是'即可逐张核对）")


if __name__ == "__main__":
    main()





