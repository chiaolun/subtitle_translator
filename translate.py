from pathlib import Path
from pprint import pformat
import textwrap

from openai import OpenAI

from tqdm import tqdm

client = OpenAI()

PROMPT = """Please translate some subtitles for me, line by line. I
will feed you one line at a time, please return only the translation
for that line, in traditional chinese.
"""


def translate_sbv(input_sbv):
    sections = [{
        "ts": s.split("\n")[0],
        "en": " ".join(s.split()[1:]),
    } for s in (
        input_sbv
        .replace("\xa0\n", " ").split("\n\n")
    )]
    prefix = [
        dict(role="system", content="You are a helpful assistant."),
        dict(role="user", content=PROMPT),
        dict(role="assistant", content="Understood, ready to proceed."),
    ]
    results = []
    for r in tqdm(sections):
        results.append(r.copy())
        messages = prefix.copy()
        for d in results[-11:]:
            messages.append(
                {"role": "user", "content": d["en"]}
            )
            if "zh-tw" in d:
                messages.append(
                    {"role": "assistant", "content": d["zh-tw"]}
                )
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.,
            messages=messages,
        )
        reply = completion.choices[0].message.to_dict()["content"]
        results[-1]["zh-tw"] = reply
        # tqdm.write(pformat(results[-1]))

    output_sbv = "\n\n".join([
        l["ts"] + "\n" + "\xa0\n".join(
            textwrap.wrap(l["zh-tw"], 40)
        ) for l in results
    ])

    return output_sbv


input_path = Path("input")
output_path = Path("output")
output_path.mkdir(exist_ok=True)
for input_file in tqdm(input_path.rglob("*.sbv")):
    output_file = output_path / input_file.relative_to(input_path)
    if output_file.exists():
        print(output_file, "exists, skipping")
        continue
    input_sbv = input_file.open().read()
    output_sbv = translate_sbv(input_sbv)
    output_file.write_text(output_sbv)
