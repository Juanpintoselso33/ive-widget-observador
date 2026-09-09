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
from widgets.seguridad.components import (
    MARCA_ABSTENCION, brecha_nacional, interpretar,
)
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
        assert MARCA_ABSTENCION in texto.lower()

    def test_intervalo_que_toca_50_tampoco_afirma(self):
        _, texto = interpretar(36, self.COLORES, (25.0, 50.0))
        assert MARCA_ABSTENCION in texto.lower(), (
            "un extremo que se muestra como 50% no puede afirmar mayoría"
        )

    def test_extremo_que_redondea_a_50_tampoco_afirma(self):
        _, texto = interpretar(36, self.COLORES, (25.0, 49.6))
        assert MARCA_ABSTENCION in texto.lower()

    def test_intervalo_claramente_de_un_lado_si_afirma(self):
        _, texto = interpretar(20, self.COLORES, (12.0, 29.0))
        assert "en contra" in texto
        _, texto = interpretar(80, self.COLORES, (71.0, 88.0))
        assert "a favor" in texto

    def test_sin_intervalo_usa_la_escala_de_siempre(self):
        _, texto = interpretar(20, self.COLORES, None)
        assert "en contra" in texto


def _modelos_entrenados():
    """
    Los JSON de producción que existan, como (slug, modelo).

    Los invariantes de esta sección valen para las CUATRO preguntas, no sólo
    para la que esté por defecto: cada una tiene su propio bootstrap, y una
    réplica desalineada en cualquiera de ellas publica un intervalo inventado.
    """
    salida = []
    for slug in config.SLUGS:
        ruta = config.ruta_modelo(slug)
        if ruta.exists():
            with open(ruta, encoding="utf-8") as f:
                salida.append((slug, json.load(f)))
    return salida


HAY_MODELOS = any(config.ruta_modelo(s).exists() for s in config.SLUGS)
SIN_MODELOS = "todavía no se entrenó ningún modelo"


class TestIntervaloProbabilidad:
    PERFIL = dict(tramo_edad=2, es_mujer=0, nivel_educ=1, ideologia=2,
                  victima=1, es_montevideo=0)

    def test_devuelve_none_sin_bootstrap(self):
        assert intervalo_probabilidad({"coefficients": {}}, **self.PERFIL) is None

    @pytest.mark.skipif(not HAY_MODELOS, reason=SIN_MODELOS)
    def test_sobre_los_modelos_reales(self):
        for slug, modelo in _modelos_entrenados():
            bajo, alto = intervalo_probabilidad(modelo, **self.PERFIL)
            assert 0 <= bajo <= alto <= 100, slug
            # El orden guardado tiene que arrancar con el intercepto y seguir
            # con los predictores: si se desalinea, los coeficientes se aplican
            # a la dummy equivocada y el intervalo sale de cualquier lado.
            orden = modelo["bootstrap"]["orden"]
            assert orden[0] == "intercept", slug
            assert orden[1:] == list(config.PREDICTORES), slug
            assert all(len(fila) == len(orden)
                       for fila in modelo["bootstrap"]["replicas"]), slug


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

    @pytest.mark.skipif(not HAY_MODELOS, reason=SIN_MODELOS)
    def test_la_banda_contiene_al_intervalo_mostrado(self):
        for slug, modelo in _modelos_entrenados():
            self._comprobar_banda_contiene(slug, modelo)

    def _comprobar_banda_contiene(self, slug, modelo):
        for perfil in (self.PERFIL, self.TESTIGO):
            iv = intervalo_probabilidad(modelo, **perfil)
            bd = banda_decision(modelo, **perfil)
            assert bd[0] <= iv[0] and bd[1] >= iv[1], (
                f"[{slug}] la banda de decisión tiene que ser al menos tan "
                "ancha como el intervalo mostrado, nunca más angosta"
            )

    @pytest.mark.skipif(not HAY_MODELOS, reason=SIN_MODELOS)
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

        n_ideol = len(config.IDEOLOGIA_UI_TO_CODE)
        perfiles = [
            dict(tramo_edad=te, es_mujer=mu, nivel_educ=ed,
                 ideologia=id_, victima=vi, es_montevideo=mv)
            for te, mu, ed, id_, vi, mv in itertools.product(
                range(1, 5), (0, 1), (1, 2, 3), range(1, n_ideol + 1),
                (1, 2, 3), (0, 1))
        ]

        # El contador se acumula sobre las CUATRO preguntas y no se exige por
        # pregunta. En "humillación a los presos" el apoyo ponderado es 11%: casi
        # ningún perfil se acerca al 50, así que ahí la banda y el intervalo
        # deciden igual sin que eso indique nada roto. Lo que tiene que existir
        # es al menos un caso en alguna parte donde ensanchar cambie la
        # conclusión; si no, banda_decision() no está haciendo nada.
        distinguen = 0
        for slug, modelo in _modelos_entrenados():
            for perfil in perfiles:
                prob = predict_probability(modelo, **perfil)
                iv = intervalo_probabilidad(modelo, **perfil)
                bd = banda_decision(modelo, **perfil)

                cruza_banda = round(bd[0]) <= 50 <= round(bd[1])
                cruza_iv = round(iv[0]) <= 50 <= round(iv[1])
                _, texto = interpretar(prob, self.COLORES, iv, bd)

                if cruza_banda:
                    assert MARCA_ABSTENCION in texto.lower(), (
                        f"[{slug}] la banda {bd} cruza el 50 y el widget igual "
                        f"afirmó: {perfil}"
                    )
                if cruza_banda and not cruza_iv:
                    distinguen += 1

        assert distinguen > 0, (
            "en ningún perfil de ninguna pregunta la banda decide distinto del "
            "intervalo mostrado: o banda_decision() dejó de ensanchar, o el "
            "test quedó vacío"
        )

    def test_la_banda_manda_sobre_el_intervalo(self):
        """Sin tocar el modelo: si la banda cruza el 50, no se afirma."""
        _, texto = interpretar(20, self.COLORES, (12.0, 29.0), (11.0, 51.0))
        assert MARCA_ABSTENCION in texto.lower()

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
        assert MARCA_ABSTENCION in texto.lower(), (
            'en pantalla dice "50% a 80%": no se puede afirmar mayoría a favor'
        )

    def test_extremo_inferior_que_redondea_a_50(self):
        _, texto = interpretar(65, self.COLORES, (50.0, 80.0))
        assert MARCA_ABSTENCION in texto.lower()

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


class TestIntervaloDeLaBrecha:
    """
    La diferencia contra el promedio nacional se bootstrapea APAREADA.

    Existe porque durante un tiempo el widget comparaba el intervalo del perfil
    contra el promedio nacional tratado como un punto exacto, y el docstring
    afirmaba que eso era "conservador de un solo lado" sin demostrarlo. La
    varianza de la resta es Var(perfil) + Var(promedio) − 2·Cov, con covarianza
    positiva; según cuánto valga, el chequeo viejo podía estar afirmando DE MÁS.
    """

    @staticmethod
    def _modelo_sintetico(probs, nacional):
        """Un JSON mínimo cuyas réplicas dan exactamente `probs`."""
        from widgets.seguridad.config import PREDICTORES
        import math
        filas = []
        for p in probs:
            z = math.log(p / (100 - p))
            filas.append([z] + [0.0] * len(PREDICTORES))
        return {
            "bootstrap": {"orden": ["intercept"] + list(PREDICTORES),
                          "replicas": filas, "nacional": nacional},
            "nivel_calibrado": 95,
        }

    def test_sin_la_tasa_por_replica_devuelve_none(self):
        """Artefacto viejo: el llamador se cae al chequeo anterior, no rompe."""
        from widgets.seguridad.model import intervalo_brecha
        m = self._modelo_sintetico([40.0] * 10, None)
        del m["bootstrap"]["nacional"]
        assert intervalo_brecha(m, 1, 0, 1, 3, 0, 0) is None

    def test_la_resta_va_replica_contra_replica(self):
        """
        Es lo que hace toda la diferencia: si perfil y promedio se movieran
        juntos, la resta casi no varía aunque cada uno varíe mucho.
        """
        from widgets.seguridad.model import intervalo_brecha, intervalo_probabilidad
        probs = [30.0, 40.0, 50.0, 60.0, 70.0] * 40
        # el promedio acompaña al perfil: la resta es constante en 10
        nacional = [p - 10 for p in probs]
        m = self._modelo_sintetico(probs, nacional)
        bajo, alto = intervalo_brecha(m, 1, 0, 1, 3, 0, 0)
        assert abs(bajo - 10) < 1e-6 and abs(alto - 10) < 1e-6, (
            "la resta apareada tiene que dar 10 exacto; si da un rango ancho, "
            "se está comparando contra el promedio de todas las réplicas"
        )
        # y el intervalo del PERFIL sí es ancho: son dos cosas distintas
        ilo, ihi = intervalo_probabilidad(m, 1, 0, 1, 3, 0, 0)
        assert ihi - ilo > 20

    def test_si_los_dos_se_mueven_en_contra_la_resta_se_ensancha(self):
        """Control opuesto: covarianza negativa, la resta varía MÁS que el perfil."""
        from widgets.seguridad.model import intervalo_brecha
        probs = [30.0, 40.0, 50.0, 60.0, 70.0] * 40
        nacional = [100 - p for p in probs]          # se mueven al revés
        m = self._modelo_sintetico(probs, nacional)
        bajo, alto = intervalo_brecha(m, 1, 0, 1, 3, 0, 0)
        assert alto - bajo > 60, (alto - bajo)

    def test_la_afirmacion_usa_el_intervalo_de_la_brecha_cuando_esta(self):
        """
        Si el intervalo de la DIFERENCIA contiene el cero, no se afirma la
        diferencia — aunque el intervalo del perfil no contenga al promedio.
        """
        from widgets.seguridad.components import brecha_nacional
        # el intervalo del perfil (27-45) NO contiene al promedio (67): con la
        # regla vieja afirmaría. El de la brecha sí contiene el 0.
        texto = brecha_nacional(36, 67, (27.0, 45.0), brecha_iv=(-8.0, 4.0))
        assert MARCA_ABSTENCION in texto.lower()
        assert "este perfil está" not in texto

    def test_y_afirma_cuando_el_cero_queda_afuera(self):
        from widgets.seguridad.components import brecha_nacional
        texto = brecha_nacional(36, 67, (27.0, 45.0), brecha_iv=(-40.0, -22.0))
        assert "este perfil está" in texto
        assert "31pp por debajo" in texto

    def test_largos_distintos_no_se_truncan_en_silencio(self):
        """
        Un artefacto con la tasa nacional desalineada tiene que RECHAZARSE, no
        truncarse. Con `min()` el sobrante se descartaba callado y la resta
        quedaba contra réplicas que no eran las suyas: Codex lo mostró con un
        ejemplo donde truncar convierte una abstención en una afirmación.
        """
        from widgets.seguridad.model import intervalo_brecha
        m = self._modelo_sintetico([30.0, 70.0], [20.0, 80.0])
        completo = intervalo_brecha(m, 1, 0, 1, 3, 0, 0)
        assert completo is not None
        assert round(completo[0]) <= 0 <= round(completo[1]), completo

        m["bootstrap"]["nacional"] = [20.0]          # desalineado a propósito
        assert intervalo_brecha(m, 1, 0, 1, 3, 0, 0) is None, (
            "con largos distintos truncaba y devolvía un intervalo que no "
            "contiene el cero, o sea una afirmación inventada"
        )

    def test_el_arranque_avisa_si_la_tasa_nacional_esta_desalineada(self):
        """
        Y no alcanza con que `intervalo_brecha` devuelva None: eso hace que el
        widget se caiga al chequeo viejo EN SILENCIO. Tiene que gritar al
        arrancar, como el resto de los problemas del artefacto.
        """
        from widgets.seguridad.model import problemas_de_calibracion
        from widgets.seguridad import config
        slug = next(s for s in config.SLUGS if s not in config.PREGUNTAS_A_RECALIBRAR)
        m = self._modelo_sintetico([30.0, 70.0], [20.0])
        assert any("apareadas" in p for p in problemas_de_calibracion(slug, m))

    def test_respeta_el_nivel_calibrado(self):
        """
        El intervalo de la brecha usa el percentil CALIBRADO de la pregunta, no
        el 95 a secas — igual que el del perfil. Los cinco tests anteriores
        pasaban aunque se ignorara `nivel_calibrado`; lo marcó Codex.
        """
        from widgets.seguridad.model import intervalo_brecha
        probs = [float(v) for v in range(10, 90)]
        nacional = [50.0] * len(probs)
        m = self._modelo_sintetico(probs, nacional)
        m["nivel_calibrado"] = 99
        ancho99 = (lambda iv: iv[1] - iv[0])(intervalo_brecha(m, 1, 0, 1, 3, 0, 0))
        m["nivel_calibrado"] = 80
        ancho80 = (lambda iv: iv[1] - iv[0])(intervalo_brecha(m, 1, 0, 1, 3, 0, 0))
        assert ancho99 > ancho80 + 5, (ancho99, ancho80)

    def test_la_comparacion_con_cero_es_inclusiva_y_redondeada(self):
        """
        Si el intervalo de la brecha redondeado toca el cero, no se afirma. Con
        una comparación exclusiva o sin redondeo, un intervalo de -0,4 a 12,3
        afirmaría — y en pantalla la brecha dice "0pp".
        """
        from widgets.seguridad.components import brecha_nacional
        assert MARCA_ABSTENCION in brecha_nacional(
            60, 55, (40.0, 70.0), brecha_iv=(-0.4, 12.3))
        assert MARCA_ABSTENCION in brecha_nacional(
            60, 55, (40.0, 70.0), brecha_iv=(0.0, 12.3))
