import base64
import glob
import io
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template_string, request
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO, YOLOWorld

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

MODEL = None
MODEL_PATH = None
MODEL_ERROR = None
MODEL_SOURCE = None

# Precomputed OpenAI CLIP ViT-B/32 embeddings for:
# cucumber, green cucumber, cucumber fruit. Keeping these tiny features in the
# app avoids loading the 338 MB CLIP text encoder on Render's small instance.
CUCUMBER_CLASSES = ["cucumber", "green cucumber", "cucumber fruit"]
CUCUMBER_TEXT_FEATURES_B64 = "iLRSOxubdbxLc1498RLtPC3cwDv7UHe9iCUqvfRS+70CQre8AiMBPfrThzxmU7m7Tozbu8vQTr1F/Kq8A869uqdNBz1k0aA7xZvYPNzaBb3BdOA8iejMO30fQjpRkT48RdK9vCyjXjqI3r+8ARZlPOCtT71T+do8WLTVOqU/pLzVLCc80rgPPTQGLj1hUGy8cfiTPDJ417xUROy8NSSnugRgqLxG9xm8ek3JPLGkKjw/AR685A3APKd81ry+sbM8r9wDvAgw6ToQnGq874t0vaMXlzyNn7w74TLXPGmbGT1/h2a8ez9TO6fhPr3B9OU85NkVPWeESrzpA8+8/xq4vANsqTs6B2a80GaWOyWgFzsITiq9XnwPPIJV3jwBwO282+aVO/rn5DtPpx+9r7rdvBPmxzxG/mi8dfOaOnkaSL3Yms88Xy3EPCnCnrwB5ps8Jf0CPOlhmry0+kQ8ZcotvN8rjjwHpYG7Q6fAuhJhgbx0EyG+yLBCO6jUx7y6YvO7kldzPYZeDj0qDRe9XHJyO9rvOz0iAhU9L3yHPGyhXL1Uuoe9ETobPdf7ODwZmTm8SVMNvK8P47t6pQC8HTWYPO1fqbwDCQA9bPmVPG0Tsrxbe8w8wk9ju25ZkzwzGN48BOTGPAU2nrzleJa8ybipvDPQObwa0x28SuETO+WBULzQCKQ8y3QSPSL4hjxL7j08ToCfvCcaDD8Sat47SK2Nu+KxY71QllC9KVCUPLNODb0Xn8W55KyaPFzUozzuLyc7GOpIvHml1rxt9sc7QY90PO3Vt7w5DVO8tZ4ovD/1Sj26WTQ8lCcGO9naCbyX3DU7JGFRvGjXJjwwoCg8VnPjOGBYjzyKVr+6uhnfPDYtaLyCoZi8+fwqPVtmGb2PD1u72ZiUO1mZ8LvpAtO8LQpEPWhFvjqxZZK8CPcgvULsPrzJnFG82d2QPPjhjzyLoxe9K/+UPBNEPjy30jQ8Fq4wvOlNxDx8Fe47q+xBPN4agbwEacE8AmGjPATXvDzT/rW88+FXu7BMDzzVnJq3pI4ZvKmbsLxBFj28T3SJO1+2GrzTfmA8SnqzO/TkbjwLkRs905MNPNASCzwE+3g6jK3gu0ztFL3T9Ji7yTYVPTkBHD1Hbte72ZjgvEwA3Tw6kxy7po9svdDF5rtELYu8psajvEQLhjvgfIK7l8mRvPdIprxEYqy7onWauijyKL0vtVw6lUKzu7nEVrx4n0G8K575PAdWS7yebSe9thU+N/bVNj3qogy9NNx8vPNLETuq46E8R7pGOw2XHD3usW09k40nPUu1tjyD3yk8cqprPJfZvDqAx9K7ykEnOUXAnby4NT27rLYYPc1mJT3Ak7S77AvxPNCf8bsxF+G7C4/COkJt4rs1lNi7cIGdvJRvmjwBea+7m9CGPGuQE7z2P0M762l1vZjpVjxVMJW7tGi6O0cJEr31P1C8S/Wvu8G40LvLDUy7vyk2vOxAP7y0Tx28agrDvD5dkD1tWsq8u/aUPKvRQD0w20S81w4AvTIF1DxeK8a8zuSuu+Yz7byIeRK89hFyO8v0prvEEf+7WXAHvRspbzxJT7k75iGNvMBsZjxmy0M9jBxjvIYrkrxH4wE8ZN9Ru79Y2bvRmq08U6UVvHc6E7w8+QA9SG6mPI4+6bnu6KS8UecLP52LjjuKAiS99uIPPT34tjw4d/U7PKXHPOkVSzwK2bm8Y141PS2yDL2Md0S8NzrivBc1pbsOWXu6P47hPK4ppTwKMnW+wH5kPMq7JTxOVnm77GLbPKaFJT3NYh46v90EPZSkGjx+6Ie63BfDvDVWObztQzs6d5htPLSj2Dv7r1Y7uKUbvDNWlrypAE68JnUovU1wibxH2xU9tn0aPErNgDzZ1aU8G8o0PfF+gzyWzQ87fvCZu3jm7DuQkIa8ppaavGxtATtNiYI9dxETvJ1wCTxN+0g7EtAiu3ZkW7wdXNM7VMRNPUUlRjx5EoE8WEGbvHoUeruWEKS7rNoGPHi9UbzIgbS8zxgivWz8+Dup8Ra9mdOWvF5MSTzBBWO8EJxlvIgngrxHKDQ9qx4QPd1CWby2om67n96evB/c8Ltihvi8ve6YOxKsFT2lAfI87go8vKdTkrxmd8u8o057PYnwarw9Isu89RkMvDLTkjuC6Fy9xB9pvDdSAD2KhIA8vzC2PJ1hjb161/27V1ICu7NOvLyV65w7MxilPOkppDy561k59SgdPVshFr1NdA07QaThvH7UX7ydbP08PY42vWhAJTzOyY87opmnu19vcb1Eu7c7ms+aPOBNbrzdHjK8ZQWuO17Ilzwl6Qg6Vw7yOhiBmLyRPxc75+FMvMUzwzxRTk+73SkvvKyQoDuHrKc8sa5lOwpDWDyJJLw8raUSvQ15mT01qDk9h0Y8vMOcVLsx7nQ8ZRjcPGRr7bx/PIa6vZl3PUteqTxMgDk8kWI3OgYZizt9VGY8Bn7TvO+dHDpEgyM9DlOvuwSqLT1GtAe9irbdu3pHUDxvp+07sBoMvTpsG7wvtYK8IvcGPYhrfbxNsrE8sIhYPZqt8Dw5Rpc8gD9hvM+ZML1UoAI9rp+PvYhSm7z19k+8qIXVusWm/btUHgU9aT4wvc8Ti7ysRgE91T6LvOlGij3dHXM9d6agPNA12LtrcJw8kZ05u6D8Iz0ojdG8q+4Yvbg0Pz1lntw7SM7KPDF/cLwoJLI8ODsVvTvB8DyjkRS833sJu0+kSzy3Xco7s1o8vPxd6jxu9Mc8qrnVO5TblL0IppW91xfHvY9nCL0f4Xi7lT4VPYDwTry1pCe8qu8avfylILvlQ5o8UbQbPT2uvbumsyc8FhuovAjE2TwdvwI8uTYsvH2iJTsoPqe89VZsPKwhzLzDRIs8U85GvT6YDD37J/C7q44IvD+tQrol9wY9klNlPdhGVbzq5x49E5NlvAxy5LvJEjq7fyVhvPq/pjtyxIM8lC0RPIbhW7ziJaM83tZkvbn0uzwYjWY8SP3vO6jzEL02Ism90fuwPAP8xTx3/z08QhEwPTeULrwvp2W8oJCZvMBCcTwoTYY8u6K8vHqtCr3PN/i8nMA8PCB07Ls2qf46Ma6UPM2CfDtq8MQ7DP3MPCIVHL3+y3G6jN1OO/UeIb0aYQy83p3IPNC++bxpF/I6M4YXvds/pDxEFWE9FpRHvewpojwC8I885L37vHI9gjyWvJS8PfTUPOZivzsmlGC8+mSBu6MjD77DzsS8k3LavBKWl7vd2A09usQqPVXWkr1JNBU8JnA5PVOFAT3byCU89mtJvUPsib3OwgQ99e9zPLy48LvgZAG8DELrOjcVtrxSEL48i5ThvLdX4Dz3ECY8ZTLqvH0LmTz3Azu8PfoSPcej6TsrqSM8MscpvXoNVztLXx69Y5d+vDitijurtVE8MpHhvA/GVDz0f0E9qJeOPPCfZLxzxqa8c/MCP+1FKDnbDFy80Dhava78gb0QvEM89gEzvSf0fLzhHmA9RzEwPOdOPLyciKu8t0F1vfe5OLyhyCY8S8d/vKMvYTqHWFQ8AwIRPbgEnzwfHqo8Ld3pvJSpgTuHXrq8fnlKPDedwztk2YC8urEEPcMKE7sSrNw82LBKvBui0rzaEn08ySHqvCxHiLtXl0w8T3G9vAp/LL2zXgs9aLuxPIPONr3DyQu9xVymOjrxlbwJZJm70+gIO7iVJ71Mw9m7FIR2PHFAAD2cbPW8eVg/u8Lumzo5aig7S2amvBScyTthMAA72chgPIg2ObwsNws96LMoOz6GvTtC2rm7uPFlvWkgtDynY1E8MgmUuwOVALx1Ebq8TYTePMj7DD0wcpg8alPdPJ1PITyhkJu7wTT1vAAllLvQvKQ8NVn4PFoGgrziVqe8bnuKPPCDlLwywkS9OjASvJfxOTxx3U+8SVxePIypyLvMkv+8cR4IvZiWJrsZqiW8Py4RvYGoNbzdMJc8WublvMUNAb0747A8cjKqvO48cL0A33i8n5OPPWW4GL2Rnc28XVDoOmD4zzzQFhC8QCY7PfuVRD3ZzDo9nmUUPNJhZzzD3BG80R2eO/QcfzooOj+77kEKvfsmGjvmRRo9zwomPe5t5bwi0W08s/kCPdAFkbpK4U882mxGvMWyDLz1laW86qQOPfNVv7tV/7Q8PSDivO+wMjzdlY+9AfCPvLUw4rtDIK47SlvDvKU2BLsUa3a8fCyyvE4IvDsFAha8oTH3vGPiETyWbPG8iKQgPU0nrbxCtz48eqxrPWA4zzuBcQ69igmUPPbXwjtC/bE8YFXavBeZFr1loiM8NnqavMOF+bwt7iO8NFGJvLufhzy7o2i7pcmMPIG+AD2fUeu8HTk4vTatHbzvsss8RdQfPFd5xTywSoK7eROlPCguVz1La5k8qbWku6A6cL0MxQI/7GvXPHl+ury8+RU9ViQBPeAfYzwW7i09wS38O/GKZL3SbHI9hRcXvW+7H7wPdwW9kFFcPJ8rhDx+o7A70VfzOWnXTb6Axw+8LCdou1w5ijw1BIg83umuPK+Qhbtnazk9f7kIPNcZ67sN0we9xQ8yvaAQB7y0m/w8oAhwvPGNsDxJhNE7GxbSvJFrrryliCm9NUGDvKABKT2kNpw8OetPO+/5Aj2flho9h/fUOvNfGbwWWlO8zLJYvDZRLb14NE68Z7rXuuX1iT31ku+8kZQtOz/tYDyLoiE6cwiku4sszzwlwus8kaklPGUH1zvBIaq7XNS2u+5x+rzNSHU8kt+2vFNbqDsDoiK9uZ65PPsE6bwUpBm9nqwFPW36tbwbpdG8qXqAOzpLAT2U4Tk97AeOvEKMALx+ZD69t/4PO/8jqrxuZt88XayEPAeY/Ty01qK8hU/kO87YpLxomYw9LjRkvRQEyrwn5iC8Hg8OvD/LHb3DTVG8ED5gvDgjhrubx2A5erKrvc0U87xWZik8p/wTu06PjrzsYtg8m5ubO2nZg7yvKxo9N2VivWERe7zJnsy8TF2qvIjkOT3l06q9fV6MPHCo/jz5JXk7/4KbvSUcITsUuxA8Y+hXvFMLKjqEyM87XUYEPWJDZzwcbLo829wFvQz7KTsf+qU8mTCHPEWZQzwJH9G8qgjju+HMhTxFoAc7WuM2PLRdGD0RcZm8NeQPPYxzKT38yHS8ZPduO8oRAjxc7s47Xh8nvTXiELz0m5A9b8ciPUHZzjy1NzU8U/dNvP311TtPMQy9YhrlPLzfiz2e+tW825iPPUWnOr2DVXi8xceFPKtPmDysZya9VrXuvEv6pTy78uE8hl6hu3z0JLymcwk9AMYIPWp5IrxWLQQ83qZRvYmhqDyOym69GO7EvIPdCr0oVnw8G3dzPNmHKzxfXhS9J+ZSPBsmRTrDWTa8i770PfsKaD2rid47thUDvEOE2zyova66VB3NPL4QEb0G3Qe91g1iPet2tTw4Dnc9b7Xtu0eX2TzeuxO9h9g6PXeHDD0r7pS85znOPLatpDr4a7K8IsAlPfzVBT1wl8+8ImI9vUuTGL3Xfam96ZDUvIPgSTywjuU8tgfNOk5ChLy94iq91XEjvQX1hjuKjBw9HuNCPA23vbvdG+e8V0pbPLyYljz/lpi82BGRPFTP2rwCCwo9ATuCvAykmzz6KA29jYgnPT8NpLygVC47CFvbvGomJT3S7ca6sr2bvMYQLz0/dQI82ktRvLTqq7uFZei8XMyVPDEJSj0H6bs7oaTLu0aPVTwPHaO8P72qPFwqmbyqiD28V8Q6uQT7UL2py5A8Wx7tO8EpBj1UGgM9R6PqvOVakjtSGyi9HjYWPRX38Tyjb1e9B6AZvRbhCL2kwRU7+YzlupqCf7wcmVg9RgzWvJkbA7vAc+88QPdBvUV9zztetkQ8oOwyvb4Z6Ls7ZZ08UcR8vNmHary5hmS9nbfSPLycJz0RLkS8lgJ2OrpidDsLKC295pW+Ox/ew7zM7eI8XEWRvPhKVLzPyuS7TTz0vTvXnTxRh8C8HdsCvS71HD2gWM88bfJMvcPoDj2MLUA9UyAQPc8TGj2o+2u9oa9+vfTzLj38/4U7qvuQvBvaijxnjCS8tQ2tvCpCC7vz6TK7+NkwPQYQrTxVdc286vjGPOcFILxeaYQ8O2/IPFAU/Txg/Ga9IJPxuiDAfr0hO4q8zrDCvIsIzDyOXhm9Mhy/PMPLpjypSXM83NJkPOGVhrxodwU/J228OwVI/rr3Z4O9EtCDvagFwTutI0u81tQbPItDUz2t3688oHuVPP/XErwdsC+9U3miPGdSm7wDwIa8VBj3vEw10byFj0I9dnGdO2K8njsGFei8eT2rO/EX1LxyLns7to0zPEuerryhBI88b/zjvNdhxbu5UuA78tCWvGnQFj2mWby8UNFzvExCoLz3jAO89zKCvGAjXz2oP6I8DZd0vcwmLb0T0oQ8mnhOvEy0xTzvpLM83C5AvYckVLzwLJE8HvhmvGt127lrBCI5vKO9O6TpqjxQC+28Lb/8PDO52Dy/BBo9UJ2Kur95gTzqdu87AdmtPKuSEL1Rmhe9JmlqvHB2azzanLg7+z7Yu5xhdbzCaDI8OW/ePFHa2zuh/9U8WT6du4TCfbzNRjy9EnwdvDdpIT0mIi09m05NuqPx+7yUDI08UZqGvJ3JVb2kc387XFd0u3puKrykvoM84auwO+AMSrwIRim94YiIu3UdBrwOiIS9FlIFPElhOzqpqA69Q4+GvC6O/Dy7Kwo8c0IJvWYPlbxbvXY9nyQpvaZchbxKwkG8m1vVPC8rjbzoEBs9OMbKPIOnFz1zfxc7L1v2ulKqrTzPZ+k8zKY3vVBs0jvHdIi8M5CnvE24Kj10Bhs9W9mxuhwX1zyxxlc87lJNvMoHnTsHvRA9ntK5vJI6qDrIGzk9Z41pu+f2Rjyrqo280ECJPOnImb2NxNY7v9OJvHt2Zjx7ie282+nvvMJ1W7wS6gO9/BCzO5MnSrt+Y727Dd27vIKktbx5dKs9QBwCvSf2gzzfukU9S3dyvINNDb2Os/g7yD98ursuoDxGHy29ecibvIWvzDz7Ic28QQvbvBYoo7zL7KQ8OCu/OyB95LsZVQu8uG2PPBrq27yH3wO9+udQPPRiF7zz9dA75ZEkPMthu7whICY8FB43PaOnezyLITy8OV0PvYZhBT9Fxzi80oEgvTEiEz2/NvQ8jOwpO3YdBD0yvZm7B5DpvH3ugT1sWBW9wbpBvWsJDb0rz3m8YDeGPLAJEz1ak4k8C1YgvrzqrDvbgS879KWRvI+tgDzdFS49C2odPOcJMj2PcZi89HiQvCuarLwf3gO94lShOxPnoTwA1Bo7SBeTvL0Vgrtjep68e366vPztu7zPuBG9lD4YPYzJjDyxDo+6g6JAPVXEBj3PipW8+FjeOOZ3ZLx0MT88Er4FvcVhtDrLyN+8Hh4mPeXkLbyMnIq8sOEFPHWQAbzb/q67XfqvPCWOmj3em5g7rPQNPKXfCjqIgIW7j9KDvNy8HDxMhoq7hAIFvTwJxbzXE7c7vt/0vEovNr0Opjo8U9FovEc4Jr342lW7ak1xPdaZLD0LnIS7T9sqvHDZLb0AbVM7LxfnvLKRpTyabxQ9MtxHPYZmqbzIBcu8SDqbvNAwjz1ZLnK8yCvuvMK1IruV55I8PXdrvfT3trzy4wm7gpxYPAac7zxyHbO9/byFvEu7ZzyTWiW901KlvCy95jyngwg9jS8VPIHuFz0t1iG9bKZMPJIYJr3JRpy8joEDPYrkkL2G+gw9v7EyvFoLd7yNaOy8cR4nvBigRjxK87i8EgaFuVtBwzw/EZc8o/6fvP6+vjx9A1G8KeVTPESXI7t31jo9YvODulFn07ywnOI58XCePNAVlLy2pPE8PENTPZXPOrzcFXE9vyUKPTzqnbzeKB68Br22PHS6ITxx6/u8gh9rPPz4TT0FPIg8J/IgPBMTBbwkM9m8gIxePE72srxzr7E7sodZPd47N7wsXUo98F03vZ4LDrwB6Qg81wBzvG0qVb3sZ3Q7gqZhuo55uDx4hqS7V13fOx/1Cj2oa4k8qxIEPS3DQTzBHe28mtWYPNUlWL3kJ5m862icu+SYczyLSQu8YVwIPTgvQr3cd9o70ReoPBG4Gry54/M9XEmiPSyxMzwbciW7d4IsPYTyqbxYWUc9DExVOpJa67yEUUE9zJPYPO+fRT3d4ie8KN28PESzobw82kY9mnkNPcaEDbypkkA8"


def discover_model():
    preferred = ["object_detection/weights/best.pt", "object_detection/weights/last.pt"]
    for path in preferred:
        if Path(path).is_file():
            return path
    candidates = glob.glob("**/*.pt", recursive=True)
    if not candidates:
        return None
    candidates.sort(key=lambda path: (Path(path).name != "best.pt", len(path), path))
    return candidates[0]


def get_model():
    global MODEL, MODEL_PATH, MODEL_ERROR, MODEL_SOURCE
    if MODEL is not None:
        return MODEL
    MODEL_PATH = discover_model()
    try:
        if MODEL_PATH:
            MODEL = YOLO(MODEL_PATH)
            MODEL_SOURCE = "A projekt saját uborkamodellje"
        else:
            # The upstream README lists custom weights, but they are not actually
            # present in the repository. YOLO-World gives us a real, usable
            # open-vocabulary fallback instead of a non-functional demo.
            MODEL_PATH = "yolov8s-worldv2.pt"
            MODEL = YOLOWorld(MODEL_PATH)
            import torch

            features = np.frombuffer(
                base64.b64decode(CUCUMBER_TEXT_FEATURES_B64), dtype=np.float32
            ).copy()
            MODEL.model.txt_feats = torch.from_numpy(features).reshape(
                1, len(CUCUMBER_CLASSES), 512
            )
            MODEL.model.model[-1].nc = len(CUCUMBER_CLASSES)
            MODEL.model.names = CUCUMBER_CLASSES
            MODEL_SOURCE = "YOLO-World előre számított uborka-szövegjellemzőkkel"
        return MODEL
    except Exception as exc:
        MODEL_ERROR = f"Modellbetöltési hiba: {exc}"
        raise


def dashed_line(draw, points, fill, width=4, dash=12):
    for start, end in zip(points[::2], points[1::2]):
        draw.line([start, end], fill=fill, width=width)


def dashed_rectangle(draw, box, fill=(255, 145, 0), width=4, dash=13):
    x1, y1, x2, y2 = [int(v) for v in box]
    top = [(x, y1) for x in range(x1, x2 + dash, dash)]
    bottom = [(x, y2) for x in range(x1, x2 + dash, dash)]
    left = [(x1, y) for y in range(y1, y2 + dash, dash)]
    right = [(x2, y) for y in range(y1, y2 + dash, dash)]
    dashed_line(draw, top, fill, width, dash)
    dashed_line(draw, bottom, fill, width, dash)
    dashed_line(draw, left, fill, width, dash)
    dashed_line(draw, right, fill, width, dash)


def estimated_full_box(box, image_size, visible_percent):
    """Symmetric geometric hypothesis, deliberately labelled as an estimate."""
    x1, y1, x2, y2 = [float(v) for v in box]
    image_w, image_h = image_size
    box_w, box_h = max(2.0, x2 - x1), max(2.0, y2 - y1)
    factor = min(10.0, max(1.0, 100.0 / visible_percent))
    if box_h >= box_w:
        new_h = min(image_h * 0.95, box_h * factor)
        center_y = (y1 + y2) / 2
        return (x1, max(0, center_y - new_h / 2), x2, min(image_h - 1, center_y + new_h / 2))
    new_w = min(image_w * 0.95, box_w * factor)
    center_x = (x1 + x2) / 2
    return (max(0, center_x - new_w / 2), y1, min(image_w - 1, center_x + new_w / 2), y2)


def annotate(image, result, visible_percent):
    out = image.copy()
    draw = ImageDraw.Draw(out)
    confirmed = 0
    suspected = 0
    boxes = [] if result.boxes is None else result.boxes
    for detection in boxes:
        xyxy = detection.xyxy[0].cpu().tolist()
        confidence = float(detection.conf[0].cpu())
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        if confidence >= 0.25:
            color = (25, 190, 85)
            label = f"BIZTOSABB {confidence:.0%}"
            confirmed += 1
        else:
            color = (255, 145, 0)
            label = f"GYANUS {confidence:.0%}"
            suspected += 1
        draw.rectangle((x1, y1, x2, y2), outline=color, width=5)
        draw.rectangle((x1, max(0, y1 - 26), x1 + 170, y1), fill=color)
        draw.text((x1 + 5, max(0, y1 - 23)), label, fill="white")

        estimate = estimated_full_box(xyxy, out.size, visible_percent)
        dashed_rectangle(draw, estimate, fill=(255, 145, 0), width=4)
        ex1, ey1, _, _ = [int(v) for v in estimate]
        draw.text((ex1 + 5, min(out.height - 22, ey1 + 5)), "BECSULT REJTETT ALAK", fill=(255, 145, 0))
    return out, confirmed, suspected


PAGE = """
<!doctype html><html lang="hu"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rejtett uborka – gyors AI-teszt</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:900px;margin:24px auto;padding:0 14px;background:#f3f6f2;color:#17211a}.card{background:#fff;border-radius:16px;padding:22px;box-shadow:0 3px 18px #0001;margin-bottom:16px}h1{margin:0 0 8px;font-size:clamp(24px,6vw,38px)}input{display:block;margin:14px 0;width:100%}input[type=range]{accent-color:#16834b}button{border:0;border-radius:12px;padding:14px 18px;font-size:17px;font-weight:700;background:#16834b;color:#fff;width:100%}img{width:100%;border-radius:12px;margin-top:14px}.muted{color:#5e6b61}.legend{display:grid;gap:8px;margin-top:12px}.green,.orange{padding:9px;border-radius:9px}.green{background:#e7f8ed}.orange{background:#fff1dc}.err{background:#ffecec;color:#8c1b1b}.num{font-size:22px;font-weight:800}code{word-break:break-all}
</style></head><body>
<div class="card"><h1>🥒 Rejtett uborka teszt</h1>
<p>Fotózd le a fóliában úgy, hogy az uborkából akár csak egy kis rész látszódjon ki a levél mögül.</p>
<form method="post" enctype="multipart/form-data">
<input type="file" name="image" accept="image/*" capture="environment" required>
<label><b>Becsült látható rész: <span id="pct">10</span>%</b></label>
<input type="range" name="visible_percent" min="5" max="50" value="10" step="5" oninput="pct.textContent=this.value">
<button type="submit">10%-os felismerés indítása</button></form>
<p class="muted">Az első futás Renderen 30–90 másodperc is lehet.</p></div>
{% if error %}<div class="card err"><b>Hiba:</b> {{ error }}</div>{% endif %}
{% if result %}<div class="card"><div><span class="num">{{ total }}</span> lehetséges uborka</div>
<div class="legend"><div class="green">🟩 Biztosabb találat: {{ confirmed }}</div><div class="orange">🟧 Gyenge találat / szaggatott becslés: {{ suspected }}. A szaggatott alak hipotézis, nem a levélen való átlátás.</div></div>
<img src="data:image/jpeg;base64,{{ result }}" alt="Felismert és becsült uborkák"></div>{% endif %}
<div class="card muted"><b>Fontos:</b> ez most gyors prototípus. A 10%-os találatok pontosságához később valódi, részben takart uborkás fotókkal kell finomhangolni az amodális modellt.<br><small>Modell: <code>{{ model_path or 'betöltéskor jelenik meg' }}</code></small></div>
</body></html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(PAGE, error=MODEL_ERROR, result=None, model_path=MODEL_PATH, model_source=MODEL_SOURCE)
    upload = request.files.get("image")
    if not upload or not upload.filename:
        return render_template_string(PAGE, error="Válassz ki egy képet.", result=None), 400
    try:
        visible_percent = max(5, min(50, int(request.form.get("visible_percent", "10"))))
        image = Image.open(upload.stream).convert("RGB")
        image.thumbnail((960, 960))
        model = get_model()
        # Keep memory below Render's small-instance limit. Test-time augmentation
        # at 960 px tripled memory and restarted the worker.
        results = model.predict(source=np.asarray(image), conf=0.03, iou=0.45, imgsz=640, augment=False, device="cpu", verbose=False)
        annotated, confirmed, suspected = annotate(image, results[0], visible_percent)
        buf = io.BytesIO(); annotated.save(buf, format="JPEG", quality=91)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return render_template_string(PAGE, error=None, result=encoded, total=confirmed + suspected, confirmed=confirmed, suspected=suspected, model_path=MODEL_PATH, model_source=MODEL_SOURCE)
    except Exception as exc:
        return render_template_string(PAGE, error=str(exc), result=None, model_path=MODEL_PATH), 500


@app.get("/health")
def health():
    return jsonify(status="ok", model_candidate=discover_model() or "yolov8s-worldv2.pt", model_loaded=MODEL is not None, model_source=MODEL_SOURCE, model_error=MODEL_ERROR)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
