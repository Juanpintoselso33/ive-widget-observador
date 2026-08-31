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
from widgets.seguridad.model import (
    _percentil, banda_decision, intervalo_probabilidad, predict_probability,
)


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
                  victima=1, es_montevideo=0)

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


class TestBandaDeDecision:
    """
    El intervalo que se muestra y el que decide sobre el 50% son distintos a
    propósito. Ver el docstring de model.banda_decision(): el extremo del
    intervalo está simulado, y su propio error basta para dar vuelta una regla
    binaria en decenas de perfiles.
    """

    PERFIL = dict(tramo_edad=2, es_mujer=0, nivel_educ=1, ideologia=2,
                  victima=1, es_montevideo=0)

    # 18-29, mujer, terciaria incompleta, izquierda extrema, víctima sin
    # violencia, interior. Codex encontró la versión previa de este perfil
    # remuestreando las réplicas: se mostraba como 15%-49% y el widget afirmaba
    # "la amplia mayoría está en contra", pero en 456 de 1.000 corridas
    # simuladas ese extremo llegaba a 50. El modelo cambió (seis tramos
    # ideológicos, sin balotaje), así que los números concretos son otros; lo
    # que el test fija es la REGLA, no aquel intervalo.
    TESTIGO = dict(tramo_edad=1, es_mujer=1, nivel_educ=2, ideologia=1,
                   victima=2, es_montevideo=0)

    COLORES = {"primary": "#000", "text_muted": "#888"}

    def test_devuelve_none_sin_bootstrap(self):
        assert banda_decision({"coefficients": {}}, **self.PERFIL) is None

    @pytest.mark.skipif(not config.MODEL_COEFFICIENTS_PATH.exists(),
                        reason="El modelo todavía no fue entrenado")
    def test_la_banda_contiene_al_intervalo_mostrado(self):
        with open(config.MODEL_COEFFICIENTS_PATH, encoding="utf-8") as f:
            modelo = json.load(f)
        for perfil in (self.PERFIL, self.TESTIGO):
            iv = intervalo_probabilidad(modelo, **perfil)
            bd = banda_decision(modelo, **perfil)
            assert bd[0] <= iv[0] and bd[1] >= iv[1], (
                "la banda de decisión tiene que ser al menos tan ancha como el "
                "intervalo mostrado, nunca más angosta"
            )

    @pytest.mark.skipif(not config.MODEL_COEFFICIENTS_PATH.exists(),
                        reason="El modelo todavía no fue entrenado")
    def test_ningun_perfil_afirma_mayoria_si_la_banda_cruza_el_50(self):
        """
        El test que fija el arreglo, escrito como INVARIANTE y no como un caso
        puntual: la versión anterior clavaba el perfil que había encontrado
        Codex con sus números exactos, y al cambiar el modelo (seis tramos
        ideológicos, sin balotaje) el test se puso rojo sin que hubiera ninguna
        regresión — el perfil seguía bien, los números eran otros.

        Recorre TODOS los perfiles elegibles y comprueba dos cosas:
          1. si la banda cruza el 50, el texto es el prudente;
          2. existe al menos un perfil donde el intervalo mostrado NO cruza el
             50 pero la banda SÍ. Sin esa segunda parte el test pasaría
             igual con banda_decision() devolviendo el intervalo tal cual, o
             sea sin el arreglo.
        """
        import itertools
        with open(config.MODEL_COEFFICIENTS_PATH, encoding="utf-8") as f:
            modelo = json.load(f)

        n_ideol = len(config.IDEOLOGIA_UI_TO_CODE)
        distinguen = 0
        for te, mu, ed, id_, vi, mv in itertools.product(
                range(1, 5), (0, 1), (1, 2, 3), range(1, n_ideol + 1),
                (1, 2, 3), (0, 1)):
            perfil = dict(tramo_edad=te, es_mujer=mu, nivel_educ=ed,
                          ideologia=id_, victima=vi, es_montevideo=mv)
            prob = predict_probability(modelo, **perfil)
            iv = intervalo_probabilidad(modelo, **perfil)
            bd = banda_decision(modelo, **perfil)

            cruza_banda = round(bd[0]) <= 50 <= round(bd[1])
            cruza_iv = round(iv[0]) <= 50 <= round(iv[1])
            _, texto = interpretar(prob, self.COLORES, iv, bd)

            if cruza_banda:
                assert "no permite afirmar" in texto, (
                    f"la banda {bd} cruza el 50 y el widget igual afirmó: {perfil}"
                )
            if cruza_banda and not cruza_iv:
                distinguen += 1

        assert distinguen > 0, (
            "en ningún perfil la banda decide distinto del intervalo mostrado: "
            "o banda_decision() dejó de ensanchar, o el test quedó vacío"
        )

    def test_la_banda_manda_sobre_el_intervalo(self):
        """Sin tocar el modelo: si la banda cruza el 50, no se afirma."""
        _, texto = interpretar(20, self.COLORES, (12.0, 29.0), (11.0, 51.0))
        assert "no permite afirmar" in texto

    def test_sin_banda_decide_el_intervalo(self):
        """Compatibilidad: el llamado viejo de dos argumentos sigue andando."""
        _, texto = interpretar(20, self.COLORES, (12.0, 29.0))
        assert "en contra" in texto


class TestBordeSimetricoDelCincuenta:
    """
    La regla tiene que suprimir la afirmación en los DOS bordes. El extremo
    inferior no estaba cubierto: una mutación que redondeara sólo el superior
    pasaba todos los tests.
    """

    COLORES = {"primary": "#000", "text_muted": "#888"}

    def test_extremo_inferior_que_se_muestra_como_50(self):
        _, texto = interpretar(65, self.COLORES, (50.4, 80.0))
        assert "no permite afirmar" in texto, (
            'en pantalla dice "50% a 80%": no se puede afirmar mayoría a favor'
        )

    def test_extremo_inferior_que_redondea_a_50(self):
        _, texto = interpretar(65, self.COLORES, (50.0, 80.0))
        assert "no permite afirmar" in texto

    def test_extremo_inferior_apenas_por_encima_si_afirma(self):
        _, texto = interpretar(65, self.COLORES, (50.6, 80.0))
        assert "a favor" in texto


class TestPercentilValidaQ:
    def test_q_fuera_de_rango(self):
        for q in (-0.01, 1.01, 2.0):
            with pytest.raises(ValueError, match="entre 0 y 1"):
                _percentil([1, 2, 3], q)

    def test_los_extremos_exactos_son_validos(self):
        assert _percentil([1, 2, 3], 0.0) == 1
        assert _percentil([1, 2, 3], 1.0) == 3
