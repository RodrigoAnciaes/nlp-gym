#!/usr/bin/env python3
import sys
import shutil
import subprocess
from pathlib import Path

def generate_pipreqs_requirements(project_root: Path, output_file: Path):
    """
    Gera um requirements.txt com base nos imports do seu código,
    usando o executável pipreqs (que deve estar no PATH).
    """
    pipreqs_cmd = shutil.which("pipreqs")
    if not pipreqs_cmd:
        print(
            "❌ Não encontrei o comando 'pipreqs'.\n"
            "Instale com:\n"
            "    pip install pipreqs\n"
            "e certifique-se de que o executável 'pipreqs' está no seu PATH.",
            file=sys.stderr
        )
        sys.exit(1)

    cmd = [
        pipreqs_cmd,
        project_root.as_posix(),
        "--force",                    # sobrescreve requirements.txt existente
        "--savepath", output_file.as_posix()
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print("❌ Erro ao executar pipreqs:\n", proc.stderr, file=sys.stderr)
        sys.exit(proc.returncode)

    print(f"✅ Arquivo '{output_file.name}' gerado com sucesso em:\n  {project_root}")

if __name__ == "__main__":
    root = Path(__file__).parent.resolve()
    req_file = root / "requirements.txt"
    generate_pipreqs_requirements(root, req_file)