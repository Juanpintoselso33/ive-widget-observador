"""
Entry point del deploy: sirve el widget IVE.

El Observador — encuesta Uruguay 2025/2026. Inspirado en el "Build a Voter" de
The Economist y adaptado para medir apoyo a la interrupción voluntaria del
embarazo.

ACÁ NO VA LÓGICA. Este archivo era una copia casi literal de
`widgets/ive/app.py` —mismos imports, mismo orden de render, mismos textos de
error— y las dos copias se separaron: cuando el widget de seguridad estrenó la
hoja del Figma hubo que acordarse de tocar dos entry points para una sola app, y
un cambio aplicado en uno solo no se nota hasta que alguien abre el otro.

Streamlit Cloud apunta su *Main file path* a este archivo, así que tiene que
seguir existiendo; lo que no tiene que hacer es repetir el widget. Ejecuta el
entry real con `run_name="__main__"` para que corra igual que si Streamlit lo
hubiera lanzado directamente.
"""

import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).parent / "widgets" / "ive" / "app.py"),
    run_name="__main__",
)
