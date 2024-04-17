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


def parse_sbv(txt):
    return [(
        s.split("\n")[0],
        " ".join(s.split()[1:]),
    ) for s in (
        txt.replace("\xa0\n", " ").split("\n\n")
    )]


def translate_sbv(input_sbv, output_sbv=None):
    sections = [dict(ts=ts, en=en) for ts, en in parse_sbv(input_sbv)]
    prefix = [
        dict(role="system", content="You are a helpful assistant."),
        dict(role="user", content=PROMPT),
        dict(role="assistant", content="Understood, ready to proceed."),
    ]
    results = []
    if output_sbv is not None:
        results = [
            {"ts": ts, "en": d["en"], "zh-tw": zh_tw}
            for d, (ts, zh_tw) in
            zip(sections, parse_sbv(output_sbv))
        ]
        sections = sections[len(results):]

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
            messages=messages,
        )
        reply = completion.choices[0].message.content
        results[-1]["zh-tw"] = reply
        tqdm.write(pformat(results[-1]))

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
        output_sbv = output_file.open().read()
    else:
        output_sbv = None
    input_sbv = input_file.open().read()
    output_sbv = translate_sbv(input_sbv, output_sbv=output_sbv)
    output_file.write_text(output_sbv)
