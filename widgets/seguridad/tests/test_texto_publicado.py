"""
Tests del TEXTO que el widget publica, no de la aritmética que hay detrás.

Existen porque la auditoría de Codex del 7/9/2026 encontró dos defectos que
ningún test podía ver: los dos vivían dentro de funciones que dibujan, y ninguna
suite recorría los 4.032 resultados posibles (4 preguntas x 1.008 perfiles).

  1. En 1.883 de esos 4.032, el widget afirmaba "este perfil está X pp por
     encima/debajo del promedio" con un intervalo que CONTENÍA el promedio.
  2. 65 perfiles se mostraban como "0%" sobre estimaciones de 0,176% a 0,499%.
     "0%" no es un redondeo: dice que nadie con ese perfil está a favor.

La lección no es que faltaban dos casos, es que la lógica editorial estaba
enterrada en el render. Ahora `brecha_nacional()` y `formato_pct()` son puras y
se pueden barrer.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import itertools
import json

import pytest

from widgets.seguridad import config
from widgets.seguridad.components import (
    MARCA_ABSTENCION, brecha_nacional, formato_pct, interpretar,
)
from widgets.seguridad.model import predict_probability, intervalo_probabilidad


class TestFormatoPct:
    """Redondear no puede convertirse en una afirmación que el dato no hace."""

    def test_los_valores_normales_se_redondean(self):
        assert formato_pct(36.7) == "37%"
        assert formato_pct(11.0) == "11%"
        assert formato_pct(50.5) == "50%"  # redondeo bancario de Python

    def test_un_valor_chico_pero_no_nulo_no_se_muestra_como_cero(self):
        """El caso exacto que encontró Codex: 0,176% se publicaba como «0%»."""
        for valor in (0.176, 0.499, 0.01, 0.4):
            assert formato_pct(valor) == "<1%", (
                f"{valor}% se muestra como «0%», que afirma que nadie apoya"
            )

    def test_un_valor_alto_pero_no_total_no_se_muestra_como_cien(self):
        """El mismo error dado vuelta: afirmar unanimidad."""
        for valor in (99.6, 99.99):
            assert formato_pct(valor) == ">99%"

    def test_el_cero_y_el_cien_exactos_si_se_muestran(self):
        """Si el modelo diera exactamente 0 o 100, no hay nada que suavizar."""
        assert formato_pct(0.0) == "0%"
        assert formato_pct(100.0) == "100%"


class TestBrechaNacional:
    """
    La afirmación sobre el promedio nacional tiene que respetar el intervalo,
    igual que ya lo hacía la afirmación sobre el 50%.
    """

    def test_sin_diferencia_no_afirma_nada(self):
        assert "coincide" in brecha_nacional(67, 67, (40.0, 90.0))

    def test_si_el_intervalo_contiene_al_promedio_no_afirma_la_diferencia(self):
        """
        El perfil textual que reportó Codex: muestra 55%, intervalo 27%-77%,
        promedio 67%. El punto está 12 pp abajo; el intervalo no lo sostiene.
        """
        texto = brecha_nacional(55, 67, (27.0, 77.0))
        assert MARCA_ABSTENCION in texto.lower()
        # Lo que importa es a QUIÉN se le atribuye la brecha: a la estimación,
        # no al perfil. Las dos assertions de abajo son las que lo prueban.
        assert "la estimación" in texto
        assert "12pp" in texto, "el número se sigue mostrando, es información"
        assert "este perfil está" not in texto

    def test_si_el_intervalo_no_contiene_al_promedio_si_afirma(self):
        texto = brecha_nacional(20, 67, (10.0, 35.0))
        assert "este perfil está" in texto
        assert "47pp por debajo" in texto

    def test_la_comparacion_va_sobre_el_intervalo_redondeado(self):
        """
        Es el que ve el lector. Con 66,6% el intervalo se muestra como «67%», y
        afirmar una diferencia contra un promedio de 67 contradice la pantalla.
        """
        assert MARCA_ABSTENCION in brecha_nacional(50, 67, (30.0, 66.6))

    def test_sin_intervalo_afirma_como_antes(self):
        """Compatibilidad: un modelo sin bootstrap no debería romper la tarjeta."""
        assert "este perfil está" in brecha_nacional(55, 67, None)


def _modelos_entrenados():
    salida = []
    for slug in config.SLUGS:
        ruta = config.ruta_modelo(slug)
        if ruta.exists():
            with open(ruta, encoding="utf-8") as f:
                salida.append((slug, json.load(f)))
    return salida


HAY_MODELOS = any(config.ruta_modelo(s).exists() for s in config.SLUGS)


@pytest.mark.skipif(not HAY_MODELOS, reason="todavía no se entrenó ningún modelo")
class TestBarridoDeTodosLosResultados:
    """
    Los 4.032 resultados que el widget puede mostrar, uno por uno.

    Codex tuvo que escribir este barrido a mano para encontrar los dos defectos.
    Queda acá para que la próxima vez lo encuentre la suite.
    """

    @staticmethod
    def _perfiles():
        return [
            dict(tramo_edad=te, es_mujer=mu, nivel_educ=ed, ideologia=id_,
                 victima=vi, es_montevideo=mv)
            for te, mu, ed, id_, vi, mv in itertools.product(
                sorted(set(config.EDAD_UI_TO_CODE.values())), (0, 1),
                sorted(set(config.EDUC_UI_TO_CODE.values())),
                sorted(set(config.IDEOLOGIA_UI_TO_CODE.values())),
                sorted(set(config.VICTIMA_UI_TO_CODE.values())),
                sorted(set(config.REGION_UI_TO_CODE.values())))
        ]

    def test_ningun_resultado_afirma_una_diferencia_que_el_intervalo_no_sostiene(self):
        fallos, comparados = [], 0
        for slug, modelo in _modelos_entrenados():
            nacional_r = round(modelo["prob_favor_nacional"])
            for perfil in self._perfiles():
                comparados += 1
                prob = predict_probability(modelo, **perfil)
                iv = intervalo_probabilidad(modelo, **perfil)
                texto = brecha_nacional(round(prob), nacional_r, iv)
                contiene = iv and round(iv[0]) <= nacional_r <= round(iv[1])
                if contiene and "este perfil está" in texto:
                    fallos.append((slug, perfil, round(prob), iv))
        # `assert not fallos` pasa igual con CERO comparaciones. Sin este
        # conteo, una grilla vacía —o un `_modelos_entrenados()` que no
        # encuentre nada— dejaría el barrido en verde sin haber mirado nada.
        assert comparados == 4032, f"el barrido comparó {comparados}, no 4.032"
        assert not fallos, (
            f"{len(fallos)} resultados afirman una diferencia contra el promedio "
            f"que su intervalo no sostiene. Primero: {fallos[0]}"
        )

    def test_ningun_resultado_se_publica_como_cero_o_cien_sin_serlo(self):
        fallos, comparados = [], 0
        for slug, modelo in _modelos_entrenados():
            for perfil in self._perfiles():
                comparados += 1
                prob = predict_probability(modelo, **perfil)
                mostrado = formato_pct(prob)
                if mostrado == "0%" and prob > 0:
                    fallos.append((slug, perfil, prob, mostrado))
                if mostrado == "100%" and prob < 100:
                    fallos.append((slug, perfil, prob, mostrado))
        assert comparados == 4032, f"el barrido comparó {comparados}, no 4.032"
        assert not fallos, (
            f"{len(fallos)} resultados se publican como 0% o 100% sin serlo. "
            f"Primero: {fallos[0]}"
        )

    def test_la_estimacion_siempre_cae_dentro_de_su_intervalo(self):
        """Coherencia básica que Codex verificó y conviene no perder."""
        for slug, modelo in _modelos_entrenados():
            for perfil in self._perfiles():
                prob = predict_probability(modelo, **perfil)
                iv = intervalo_probabilidad(modelo, **perfil)
                assert iv is not None, f"{slug} no trae bootstrap"
                assert iv[0] <= prob <= iv[1], (slug, perfil, prob, iv)


class TestMapaDeRecalibracion:
    """
    El mapa tiene que estar si y sólo si la pregunta lo declara, y estar sano.

    Existen porque la huella del contrato NO cubre esto: cubre la declaración,
    no el contenido del JSON. Codex sacó el mapa de una copia en memoria
    conservando la huella y pasaba todos los controles de arranque — el widget
    habría publicado sin recalibrar una pregunta que se declaró que lo necesita,
    en silencio.
    """

    @pytest.mark.parametrize("slug", config.SLUGS)
    def test_el_json_entrenado_tiene_el_mapa_que_le_corresponde(self, slug):
        from widgets.seguridad.model import problemas_de_calibracion
        ruta = config.ruta_modelo(slug)
        if not ruta.exists():
            pytest.skip(f"«{slug}» todavía no fue entrenada")
        with open(ruta, encoding="utf-8") as f:
            modelo = json.load(f)
        assert problemas_de_calibracion(slug, modelo) == []

    def test_falta_el_mapa_en_una_pregunta_declarada(self):
        from widgets.seguridad.model import problemas_de_calibracion
        slug = config.PREGUNTAS_A_RECALIBRAR[0]
        problemas = problemas_de_calibracion(slug, {"coefficients": {}})
        assert problemas and "sin recalibrar" in problemas[0]

    def test_sobra_el_mapa_en_una_pregunta_no_declarada(self):
        from widgets.seguridad.model import problemas_de_calibracion
        slug = next(s for s in config.SLUGS if s not in config.PREGUNTAS_A_RECALIBRAR)
        problemas = problemas_de_calibracion(
            slug, {"calibracion": {"grilla": [0.0, 1.0], "valores": [0.0, 1.0]}})
        assert problemas and "nadie pidió" in problemas[0]

    def test_un_mapa_no_monotono_se_rechaza(self):
        """
        Es la falla que más importa: un mapa que baja da vuelta el orden de dos
        perfiles, y el widget publicaría que un grupo apoya menos que otro
        cuando el modelo dice lo contrario.
        """
        from widgets.seguridad.model import problemas_de_calibracion
        slug = config.PREGUNTAS_A_RECALIBRAR[0]
        malo = {"calibracion": {"grilla": [0.0, 0.5, 1.0],
                                "valores": [0.0, 0.8, 0.4]}}
        problemas = problemas_de_calibracion(slug, malo)
        assert any("monótono" in p for p in problemas)

    def test_el_mapa_se_aplica_al_punto_y_a_cada_replica(self):
        """
        Si se aplicara sólo al número, el intervalo publicado dejaría de
        corresponder al porcentaje que lo encabeza. Lo marcó Codex.
        """
        from widgets.seguridad.model import predict_probability, intervalo_probabilidad
        slug = config.PREGUNTAS_A_RECALIBRAR[0]
        ruta = config.ruta_modelo(slug)
        if not ruta.exists():
            pytest.skip(f"«{slug}» todavía no fue entrenada")
        with open(ruta, encoding="utf-8") as f:
            modelo = json.load(f)
        perfil = dict(tramo_edad=2, es_mujer=0, nivel_educ=2, ideologia=4,
                      victima=1, es_montevideo=0)
        prob = predict_probability(modelo, **perfil)
        bajo, alto = intervalo_probabilidad(modelo, **perfil)
        assert bajo <= prob <= alto, (
            "el punto quedó fuera de su intervalo: señal de que el mapa se "
            "aplicó a uno y no al otro"
        )


class TestLaMarcaDeAbstencionDiscrimina:
    """
    Controles negativos para `MARCA_ABSTENCION`.

    POR QUÉ. Los doce tests que verifican que el widget se abstiene lo hacen
    preguntando `MARCA_ABSTENCION in texto`. Esa comparación es VERDADERA PARA
    CUALQUIER TEXTO si la constante quedara vacía, así que los doce pasarían sin
    probar nada — el mismo agujero que se quería evitar al dejar de asertar la
    frase literal, movido de lugar. Lo marcó Codex al revisar el commit que
    introdujo la constante.

    Ver [[test-que-fabrica-su-esperado]]: una aserción que se cumple igual con
    el conjunto vacío no es una verificación.
    """

    def test_la_marca_no_esta_vacia_y_tiene_forma_de_frase(self):
        assert MARCA_ABSTENCION, "una marca vacía hace pasar todos los tests"
        assert len(MARCA_ABSTENCION) > 15
        assert " " in MARCA_ABSTENCION, "tiene que ser una frase, no un token"

    def test_un_texto_que_SI_afirma_no_contiene_la_marca(self):
        """
        El control negativo. Sin esto, "el texto de abstención contiene la
        marca" se cumpliría igual con la marca vacía.
        """
        afirma = brecha_nacional(20, 67, (10.0, 35.0))
        assert "este perfil está" in afirma
        assert MARCA_ABSTENCION not in afirma.lower()

    def test_las_dos_frases_de_abstencion_la_comparten(self):
        mayoria = interpretar(50, {"primary": "#000"}, (30.0, 70.0))[1]
        brecha = brecha_nacional(55, 67, (27.0, 77.0))
        assert MARCA_ABSTENCION in mayoria.lower()
        assert MARCA_ABSTENCION in brecha.lower()
        # Y siguen siendo frases distintas: compartir el arranque no puede
        # significar que las dos digan lo mismo.
        assert mayoria != brecha
