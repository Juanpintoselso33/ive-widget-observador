"""
Tests de los intervalos de confianza.

Cubren las tres cosas que se pueden romper sin que se note: el cálculo del
percentil, la regla que decide si se afirma de qué lado está la mayoría, y el
comportamiento cuando no hay bootstrap.

El percentil tenía un bug real: `ordenados[int(q * n)]` corre los dos extremos
una posición hacia arriba. Con 1.000 réplicas la diferencia es chica, pero movía
el extremo mostrado en cientos de perfiles y cambiaba la decisión sobre el 50%
en varios.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json

import pytest

from widgets.seguridad import config
from widgets.seguridad.components import interpretar
from widgets.seguridad.model import _percentil, intervalo_probabilidad


class TestPercentil:
    """Contrastado contra el método tipo 7, que es el default de numpy y R."""

    def test_mediana_de_una_lista_impar(self):
        assert _percentil([1, 2, 3, 4, 5], 0.5) == 3

    def test_mediana_de_una_lista_par_interpola(self):
        assert _percentil([1, 2, 3, 4], 0.5) == pytest.approx(2.5)

    def test_extremos(self):
        datos = list(range(101))
        assert _percentil(datos, 0.0) == 0
        assert _percentil(datos, 1.0) == 100

    def test_interpola_y_no_trunca(self):
        """
        Con 0..100 y q=0,025 el tipo 7 da 2,5. La versión con int() daba 2,
        siempre corrida hacia arriba en el extremo bajo y hacia abajo en el alto.
        """
        assert _percentil(list(range(101)), 0.025) == pytest.approx(2.5)
        assert _percentil(list(range(101)), 0.975) == pytest.approx(97.5)

    def test_lista_de_un_elemento(self):
        assert _percentil([7], 0.5) == 7

    def test_lista_vacia(self):
        assert _percentil([], 0.5) is None


class TestReglaDelCincuenta:
    """
    La decisión se toma sobre los extremos REDONDEADOS, que son los que ve el
    lector, y es inclusiva: un intervalo que en pantalla dice "25% a 50%" no
    puede acompañarse de "la mayoría está en contra".
    """

    COLORES = {"primary": "#000", "text_muted": "#888"}

    def test_intervalo_que_cruza_no_afirma_mayoria(self):
        _, texto = interpretar(43, self.COLORES, (31.0, 58.0))
        assert "no permite afirmar" in texto

    def test_intervalo_que_toca_50_tampoco_afirma(self):
        _, texto = interpretar(36, self.COLORES, (25.0, 50.0))
        assert "no permite afirmar" in texto, (
            "un extremo que se muestra como 50% no puede afirmar mayoría"
        )

    def test_extremo_que_redondea_a_50_tampoco_afirma(self):
        _, texto = interpretar(36, self.COLORES, (25.0, 49.6))
        assert "no permite afirmar" in texto

    def test_intervalo_claramente_de_un_lado_si_afirma(self):
        _, texto = interpretar(20, self.COLORES, (12.0, 29.0))
        assert "en contra" in texto
        _, texto = interpretar(80, self.COLORES, (71.0, 88.0))
        assert "a favor" in texto

    def test_sin_intervalo_usa_la_escala_de_siempre(self):
        _, texto = interpretar(20, self.COLORES, None)
        assert "en contra" in texto


class TestIntervaloProbabilidad:
    PERFIL = dict(tramo_edad=2, es_mujer=0, nivel_educ=1, ideologia=2,
                  victima=1, es_montevideo=0, balotaje=0)

    def test_devuelve_none_sin_bootstrap(self):
        assert intervalo_probabilidad({"coefficients": {}}, **self.PERFIL) is None

    @pytest.mark.skipif(not config.MODEL_COEFFICIENTS_PATH.exists(),
                        reason="El modelo todavía no fue entrenado")
    def test_sobre_el_modelo_real(self):
        with open(config.MODEL_COEFFICIENTS_PATH, encoding="utf-8") as f:
            modelo = json.load(f)
        bajo, alto = intervalo_probabilidad(modelo, **self.PERFIL)
        assert 0 <= bajo <= alto <= 100
        # El orden guardado tiene que arrancar con el intercepto y seguir con
        # los predictores: si se desalinea, los coeficientes se aplican a la
        # dummy equivocada y el intervalo sale de cualquier lado.
        orden = modelo["bootstrap"]["orden"]
        assert orden[0] == "intercept"
        assert orden[1:] == list(config.PREDICTORES)
        assert all(len(fila) == len(orden) for fila in modelo["bootstrap"]["replicas"])
