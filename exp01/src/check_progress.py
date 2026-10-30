"""查看仍含 NotImplementedError 的填空函数，不执行数值实验。"""
from __future__ import annotations
import argparse
import ast
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labs", nargs="+", choices=[f"{i:02d}" for i in range(1, 9)])
    args = parser.parse_args()
    pending = 0
    for path in sorted(Path(__file__).resolve().parent.glob("lab[0-9][0-9]_*.py")):
        if args.labs and path.name[3:5] not in args.labs:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except SyntaxError as error:
            print(f"{path.name}：第 {error.lineno} 行存在语法问题：{error.msg}")
            pending += 1
            continue
        functions = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                unfinished = any(isinstance(part, ast.Raise) and isinstance(part.exc, ast.Call)
                                 and isinstance(part.exc.func, ast.Name)
                                 and part.exc.func.id == "NotImplementedError"
                                 for part in ast.walk(node))
                if unfinished:
                    functions.append(f"{node.name}（第 {node.lineno} 行）")
        if functions:
            pending += len(functions)
            print(f"{path.name}：" + "、".join(functions))
        else:
            print(f"{path.name}：未发现未填标记，请运行本实验完成数值检查。")
    print(f"尚含未填标记的函数或语法问题共 {pending} 项。")
    return 2 if pending else 0

if __name__ == "__main__":
    raise SystemExit(main())
