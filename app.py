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
seguir existiendo; lo que no tiene que hacer es repetir el widget.

Va por IMPORT y no por `runpy.run_path()`: las dos formas ejecutan el widget,
pero runpy lo corre en un `__main__` temporal que después sale de `sys.modules`,
y el watcher de Streamlit arma la lista de archivos a vigilar recorriendo
`sys.modules` — así que editar `widgets/ive/app.py` no recargaba la app. Con el
import queda registrado y se recarga como cualquier otro módulo.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from widgets.ive.app import main  # noqa: E402

main()
