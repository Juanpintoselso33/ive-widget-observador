"""
Tests de coherencia de la configuración.

Cubren sobre todo el mecanismo de preguntas parametrizadas: la idea es que
agregar, sacar o reordenar una pregunta no pueda dejar el widget en un estado
inconsistente sin que un test lo avise.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json

import pytest

from widgets.seguridad import config


def test_la_pregunta_por_defecto_existe_en_el_catalogo():
    assert config.PREGUNTA_DEFECTO in config.PREGUNTAS


def test_slugs_es_el_catalogo_completo_y_en_orden():
    assert config.SLUGS == list(config.PREGUNTAS)


def test_son_las_cuatro_preguntas_que_pidio_tomer():
    """
    Las cuatro que Tomer señaló por columna de la base etiquetada (7/9/2026):
    AC mano dura, Z cadena perpetua, Y pena de muerte, AA humillación.

    Va como test y no como comentario porque el catálogo es editable: sumar una
    quinta pregunta es legítimo, pero tiene que ser una decisión y no el
    resultado de que alguien dejara media edición a medio camino.
    """
    assert set(config.SLUGS) == {
        "politico_mano_dura", "cadena_perpetua", "pena_muerte", "humillacion_presos",
    }
    assert config.PREGUNTAS["politico_mano_dura"]["columna"].startswith("var_233")
    assert config.PREGUNTAS["cadena_perpetua"]["columna"].startswith("var_230")
    assert config.PREGUNTAS["pena_muerte"]["columna"].startswith("var_229")
    assert config.PREGUNTAS["humillacion_presos"]["columna"].startswith("var_231")


def test_toda_pregunta_declara_los_campos_que_usa_la_ui():
    for slug, pregunta in config.PREGUNTAS.items():
        for campo in ("columna", "enunciado", "etiqueta", "titulo",
                      "titulo_corto", "afirma"):
            assert campo in pregunta, f"'{slug}' no declara '{campo}'"
            assert pregunta[campo].strip(), f"'{slug}' tiene '{campo}' vacío"


def test_no_hay_dos_preguntas_apuntando_a_la_misma_columna():
    columnas = [p["columna"] for p in config.PREGUNTAS.values()]
    assert len(columnas) == len(set(columnas))


def test_las_etiquetas_del_selector_son_unicas_y_vuelven_a_su_slug():
    """
    El selector devuelve la etiqueta y app.py la traduce de vuelta a slug. Si
    dos preguntas compartieran etiqueta, una de las dos sería inalcanzable y el
    lector vería el título de una con los coeficientes de la otra.
    """
    etiquetas = [p["etiqueta"] for p in config.PREGUNTAS.values()]
    assert len(etiquetas) == len(set(etiquetas))
    for slug in config.SLUGS:
        assert config.ETIQUETA_A_SLUG[config.PREGUNTAS[slug]["etiqueta"]] == slug


def test_cada_pregunta_tiene_su_propia_ruta_de_modelo():
    rutas = [config.ruta_modelo(s) for s in config.SLUGS]
    assert len(set(rutas)) == len(rutas)


def test_la_escala_likert_cubre_los_cinco_puntos():
    assert sorted(config.LIKERT_MAP.values()) == [1, 2, 3, 4, 5]


def test_favor_contra_y_neutral_no_se_solapan():
    favor, contra = set(config.LIKERT_FAVOR), set(config.LIKERT_CONTRA)
    assert not favor & contra
    assert config.LIKERT_NEUTRAL not in favor | contra
    assert favor | contra | {config.LIKERT_NEUTRAL} == set(config.LIKERT_MAP.values())


def test_la_fuente_nombra_a_los_tres_socios_de_la_encuesta():
    """
    Tomer: "en la fuente, siempre es la encuesta de El Observador-UMAD-Ferreira".
    Es un pedido explícito del cliente sobre una pieza que se publica, así que
    no puede quedar sólo en un comentario.
    """
    for parte in ("El Observador", "UMAD", "Ferreira"):
        assert parte in config.FUENTE, f"la fuente no menciona '{parte}'"
    assert config.CREDITO.strip()


class TestTramosIdeologicos:
    """
    Los siete tramos NO son una elección de este repo: son la transcripción de
    la columna etiquetada que mandó Tomer. Si alguien mueve un borde, el modelo
    sigue entrenando y el widget sigue andando — sólo que publica categorías que
    ya no son las de la fuente.
    """

    def test_son_una_particion_de_la_escala_0_10(self):
        cubiertos = []
        for _, desde, hasta, _ in config.ESPEC_CRUDA["ideol_tramos"]:
            assert desde <= hasta
            cubiertos.extend(range(desde, hasta + 1))
        assert sorted(cubiertos) == list(range(11)), (
            "los tramos se solapan o dejan huecos en la escala"
        )

    def test_son_simetricos_alrededor_del_centro(self):
        """
        El hallazgo que el widget deja ver es que los extremos se despegan. Eso
        sólo se puede leer si "extrema izquierda" y "extrema derecha" abarcan la
        misma cantidad de valores de la escala: comparar una red ancha contra
        una angosta produciría la diferencia por construcción.
        """
        anchos = [hasta - desde + 1
                  for _, desde, hasta, _ in config.ESPEC_CRUDA["ideol_tramos"]]
        assert anchos == anchos[::-1], f"tramos asimétricos: {anchos}"

    def test_la_referencia_es_uno_de_los_tramos(self):
        nombres = {t[0] for t in config.ESPEC_CRUDA["ideol_tramos"]}
        assert config.ESPEC_CRUDA["ideol_referencia"] in nombres

    @pytest.mark.parametrize("etiqueta,esperado", [
        ("Extrema izquierda", 42),
        ("Izquierda", 162),
        ("Centroizquierda", 573),
        ("Centro", 1092),
        ("Centroderecha", 884),
        ("Derecha", 413),
        ("Extrema derecha", 131),
    ])
    def test_reproducen_la_base_etiquetada(self, etiqueta, esperado):
        """
        Los conteos que produce cada tramo sobre la base tienen que ser los de
        la columna etiquetada de Tomer, verificados caso por caso en los 3.377
        registros el 7/9/2026. Quedan clavados acá porque un borde corrido no
        rompe nada más: entrena igual, sirve igual, y publica otra cosa.
        """
        pd = pytest.importorskip("pandas")
        if not config.DATA_FILE.exists():
            pytest.skip("la base no está disponible en esta máquina")
        df = pd.read_csv(config.DATA_FILE, encoding="utf-8-sig")
        col = df["var_242 | Autoubicacion izquierda-derecha (0-10)"]
        tramo = next(t for t in config.ESPEC_CRUDA["ideol_tramos"] if t[3] == etiqueta)
        _, desde, hasta, _ = tramo
        assert int(col.between(desde, hasta).sum()) == esperado


class TestMapeosUI:
    """Cada mapeo de la UI debe cubrir los códigos que espera build_features."""

    @pytest.mark.parametrize("mapeo,esperados", [
        ("EDAD_UI_TO_CODE", {1, 2, 3, 4}),
        ("EDUC_UI_TO_CODE", {1, 2, 3}),
        ("IDEOLOGIA_UI_TO_CODE", {1, 2, 3, 4, 5, 6, 7}),
        ("VICTIMA_UI_TO_CODE", {1, 2, 3}),
        ("REGION_UI_TO_CODE", {0, 1}),
    ])
    def test_codigos_completos_y_sin_repetir(self, mapeo, esperados):
        valores = list(getattr(config, mapeo).values())
        assert len(valores) == len(set(valores)), f"{mapeo} tiene códigos repetidos"
        assert set(valores) == esperados


def test_predictores_sin_duplicados():
    assert len(config.PREDICTORES) == len(set(config.PREDICTORES))


def _modelos_entrenados():
    """Los slugs cuyo JSON existe. Vacío antes del primer entrenamiento."""
    return [s for s in config.SLUGS if config.ruta_modelo(s).exists()]


@pytest.mark.parametrize("slug", config.SLUGS)
class TestModeloEntrenado:
    """Sólo corren para las preguntas que ya tienen su JSON."""

    @pytest.fixture
    def modelo(self, slug):
        ruta = config.ruta_modelo(slug)
        if not ruta.exists():
            pytest.skip(f"«{slug}» todavía no fue entrenada")
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)

    def test_estan_todos_los_coeficientes_que_espera_la_inferencia(self, modelo, slug):
        faltan = set(config.PREDICTORES) - set(modelo["coefficients"])
        assert not faltan, f"faltan coeficientes: {faltan}"
        assert "intercept" in modelo["coefficients"]

    def test_el_modelo_de_neutralidad_tiene_los_mismos_predictores(self, modelo, slug):
        assert set(modelo["coefficients"]) == set(modelo["coefficients_neutral"])

    def test_el_json_corresponde_a_su_pregunta(self, modelo, slug):
        assert modelo["pregunta_slug"] == slug, (
            f"el JSON de «{slug}» dice ser de «{modelo['pregunta_slug']}»"
        )
        assert modelo["pregunta_columna"] == config.PREGUNTAS[slug]["columna"]

    def test_trae_los_textos_que_la_ui_lee_sin_fallback(self, modelo, slug):
        """
        components.py los lee con `model[...]`, sin `.get`: si faltara alguno la
        página reventaría con KeyError en producción.
        """
        for campo in ("pregunta_titulo", "pregunta_enunciado", "pregunta_afirma",
                      "pregunta_titulo_corto", "pregunta_etiqueta"):
            assert modelo[campo].strip()

    def test_publica_fuente_y_credito(self, modelo, slug):
        assert modelo["fuente"] == config.FUENTE
        assert modelo["credito"] == config.CREDITO

    def test_trae_el_tamanio_de_cada_tramo_ideologico(self, modelo, slug):
        """La metodología los nombra desde acá en vez de tenerlos a mano."""
        tam = modelo["tamanio_tramos_ideologicos"]
        for nombre, _, _, _ in config.ESPEC_CRUDA["ideol_tramos"]:
            assert tam[f"ideol_{nombre}"] > 0

    def test_las_tasas_publicadas_son_porcentajes(self, modelo, slug):
        assert 0 <= modelo["prob_favor_nacional"] <= 100
        assert 0 <= modelo["prob_neutral_nacional"] <= 100

    def test_los_conteos_publicados_reconcilian_con_el_n(self, modelo, slug):
        """
        La sección de metodología muestra estos números al lector: si no
        cierran contra el N de la encuesta, se publica una inconsistencia.
        """
        info = modelo["model_info"]
        assert info["n"] + info["n_excluidos"] == info["n_encuesta"]
        assert info["n_neutrales_explicitos"] + info["n_sin_respuesta"] == info["n_excluidos"]


def test_los_cuatro_modelos_salen_de_la_misma_encuesta():
    """
    Si alguien entrena dos preguntas con bases distintas, el widget compara
    números que no son comparables y nada lo delata en pantalla.
    """
    entrenados = _modelos_entrenados()
    if len(entrenados) < 2:
        pytest.skip("hace falta más de un modelo entrenado")
    ns = set()
    for slug in entrenados:
        with open(config.ruta_modelo(slug), encoding="utf-8") as f:
            ns.add(json.load(f)["model_info"]["n_encuesta"])
    assert len(ns) == 1, f"los modelos se entrenaron con bases distintas: {ns}"


class TestHuellaContrato:
    """
    La huella es lo único que impide servir un modelo entrenado con otra
    codificación. Estos tests son de mutación: cambian una pieza de la
    especificación y verifican que la huella cambie. Sin ellos, la huella puede
    quedarse corta sin que nadie se entere — que es lo que pasó con
    EDUC_COLAPSO, que estaba fuera y dejaba el mismo hash.
    """

    def test_es_estable_entre_llamadas(self):
        assert (config.huella_contrato(config.PREGUNTA_DEFECTO)
                == config.huella_contrato(config.PREGUNTA_DEFECTO))

    def test_cada_pregunta_tiene_huella_distinta(self):
        """
        Es lo que impide que copiar el JSON de una pregunta encima del de otra
        pase el chequeo de arranque.
        """
        huellas = [config.huella_contrato(s) for s in config.SLUGS]
        assert len(set(huellas)) == len(huellas)

    def test_una_pregunta_desconocida_no_produce_huella(self):
        with pytest.raises(KeyError):
            config.huella_contrato("no_existe")

    @pytest.mark.parametrize("clave,valor", [
        ("educ_colapso", {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3}),
        ("edad_cortes", [17, 34, 49, 64, 120]),
        ("ideol_tramos", [["izq_extrema", 0, 2, "Extrema izquierda"],
                          ["izquierda", 3, 3, "Izquierda"],
                          ["centroizq", 4, 4, "Centroizquierda"],
                          ["centro", 5, 5, "Centro"],
                          ["centroderecha", 6, 6, "Centroderecha"],
                          ["derecha", 7, 8, "Derecha"],
                          ["der_extrema", 9, 10, "Extrema derecha"]]),
        ("ideol_referencia", "derecha"),
        ("dpto_montevideo", 19),
        ("sexo_mujer", "F"),
    ])
    def test_cambiar_la_especificacion_cruda_cambia_la_huella(self, clave, valor, monkeypatch):
        slug = config.PREGUNTA_DEFECTO
        original = config.huella_contrato(slug)
        espec = dict(config.ESPEC_CRUDA)
        espec[clave] = valor
        monkeypatch.setattr(config, "ESPEC_CRUDA", espec)
        assert config.huella_contrato(slug) != original, (
            f"cambiar '{clave}' no cambió la huella: un JSON viejo cargaría igual"
        )

    def test_cambiar_un_mapeo_de_la_ui_cambia_la_huella(self, monkeypatch):
        slug = config.PREGUNTA_DEFECTO
        original = config.huella_contrato(slug)
        monkeypatch.setattr(config, "EDUC_UI_TO_CODE",
                            {"Primaria": 1, "Secundaria": 2, "Terciaria": 3})
        assert config.huella_contrato(slug) != original

    def test_reordenar_predictores_no_cambia_la_huella(self, monkeypatch):
        """El orden de la lista no tiene significado: no debe invalidar el JSON."""
        slug = config.PREGUNTA_DEFECTO
        original = config.huella_contrato(slug)
        monkeypatch.setattr(config, "PREDICTORES", list(reversed(config.PREDICTORES)))
        assert config.huella_contrato(slug) == original

    @pytest.mark.parametrize("slug", config.SLUGS)
    def test_el_json_entrenado_coincide_con_la_huella_actual(self, slug):
        ruta = config.ruta_modelo(slug)
        if not ruta.exists():
            pytest.skip(f"«{slug}» todavía no fue entrenada")
        with open(ruta, encoding="utf-8") as f:
            modelo = json.load(f)
        assert modelo.get("contrato") == config.huella_contrato(slug), (
            f"el JSON de «{slug}» no corresponde a la configuración actual: "
            "hay que volver a correr train_model.py"
        )


class TestNivelCalibradoContraLaMedicion:
    """
    El nivel publicado no puede ser MÁS angosto que lo que sostiene el estudio
    de cobertura.

    Existe porque el nivel es un número suelto en un dict: nada impedía bajarlo
    "porque el intervalo se ve muy ancho". Acá el criterio queda atado a las
    salidas de `cobertura_simulada.py` que viven en `scripts/salidas/`.

    Que quede MÁS ancho sí se permite. Durante un tiempo pasaba: mano dura
    publicaba 98 cuando el criterio ya se cumplía en 97. Yo había escrito en
    `config.py` que era una "decisión editorial declarada" y NO lo era —la tomé
    yo sin consultar, y al preguntarlo el 11/9/2026 Juan pidió el nivel que
    dicen las simulaciones—. Hoy las cuatro publican lo que dice el criterio.

    ESTE TEST SE DEJÓ ENGAÑAR DOS VECES, las dos encontradas por Codex con
    control negativo, y cada versión pasaba los 148 tests:

      · v1: recorría las preguntas que ENCONTRABA en los JSON, no las que hay
        que publicar. Borrando las salidas de cadena perpetua se podía bajar su
        nivel a 95.
      · v2: contaba ARCHIVOS, no corridas independientes. Reemplazando las dos
        corridas de cadena perpetua por dos copias de la semilla 402 se podía
        bajar su nivel a 98. Y aceptaba mezclar corridas con B distinto, y
        aceptaba que una salida nueva omitiera la huella.

    Ahora se exige: las cuatro preguntas, al menos dos SEMILLAS distintas por
    pregunta, el mismo B en todas las corridas de una pregunta, y huella en
    TODA salida, sin excepciones. La lista de ocho hashes históricos que eximía
    a las corridas del 8/9/2026 se borró junto con ellas: el estudio que rige
    desde el 11/9/2026 es el de B=10.000, y sus ocho salidas traen sello.

    Y LA HUELLA CAMBIÓ DE DEFINICIÓN. Era `huella_contrato`, que es el contrato
    de codificación: Codex verificó que sobrevive intacta a cambiar `C_GRID`,
    `NODOS_CALIBRACION` y `RANDOM_STATE`, y que no mira el artefacto usado como
    verdad. Ahora es `cobertura_simulada.huella_estudio`, que suma las perillas
    del procedimiento, el modelo que hace de verdad y el AST —sin docstrings—
    de las funciones que definen el pipeline.
    """

    @staticmethod
    def _modulos():
        import importlib
        import importlib.util
        from pathlib import Path as _P
        scripts = _P(__file__).parent.parent / "scripts"
        spec = importlib.util.spec_from_file_location(
            "agregar_calibracion", scripts / "agregar_calibracion.py")
        agg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agg)
        sim = importlib.import_module(
            "widgets.seguridad.scripts.cobertura_simulada")
        return agg, sim, scripts / "salidas"

    def test_ningun_nivel_publicado_queda_por_debajo_del_medido(self):
        import json as _json
        agg, sim, salidas = self._modulos()
        # NO se saltea si no hay salidas. Un nivel por encima de 95 es una
        # afirmación de que el bootstrap percentil sub-cubre y de cuánto; sin el
        # estudio no hay nada que la sostenga, y borrar la carpeta no puede ser
        # la forma de aprobar el test. Lo marcó Codex: la versión anterior hacía
        # skip y dejaba pasar cualquier nivel.
        if not list(salidas.glob("cal-*.json")):
            sin_respaldo = {s: n for s, n in config.NIVEL_CALIBRADO.items() if n > 95}
            assert not sin_respaldo, (
                f"no hay salidas en {salidas} y estas preguntas publican un "
                f"nivel por encima de 95: {sin_respaldo}"
            )
            return

        por_slug = agg._cargar()

        faltan = [s for s in config.SLUGS if s not in por_slug]
        assert not faltan, (
            f"hay salidas del estudio pero ninguna de {faltan}: el nivel de esas "
            f"preguntas no está respaldado por nada"
        )

        flacas, mezcladas, sin_sello, desfasadas = {}, {}, [], []
        for slug in config.SLUGS:
            corridas = por_slug[slug]
            semillas = {c.get("semilla") for c in corridas}
            if len(semillas) < 2:
                flacas[slug] = sorted(semillas)
            bes = {c.get("replicas") for c in corridas}
            if len(bes) > 1:
                mezcladas[slug] = sorted(bes)
            for c in corridas:
                clave = (slug, c.get("semilla"))
                huella = c.get("huella")
                if huella is None:
                    sin_sello.append(clave)
                    continue
                ruta = config.ruta_modelo(slug)
                if not ruta.exists():
                    continue
                with open(ruta, encoding="utf-8") as f:
                    modelo = _json.load(f)
                if huella != sim.huella_estudio(slug, modelo):
                    desfasadas.append(clave)

        assert not flacas, (
            f"el criterio compara corridas con semillas distintas y estas no "
            f"las tienen: {flacas}"
        )
        assert not mezcladas, (
            f"estas preguntas mezclan corridas con distinto B, así que su "
            f"promedio no es de un solo procedimiento: {mezcladas}"
        )
        assert not sin_sello, (
            f"estas salidas no traen huella del estudio: {sin_sello}"
        )
        assert not desfasadas, (
            "estas salidas se midieron con otro procedimiento o contra otra "
            f"verdad que la que se publica: {desfasadas}"
        )

        flojos = []
        for slug in config.SLUGS:
            minimo, _ = agg.elegir(por_slug[slug])
            assert minimo is not None, (
                f"«{slug}»: ningún nivel medido llega al 95% en todas las "
                f"corridas; no hay respaldo para el que se publica"
            )
            if config.NIVEL_CALIBRADO[slug] < int(minimo):
                flojos.append((slug, config.NIVEL_CALIBRADO[slug], minimo))
        assert not flojos, (
            "estos niveles publicados son más angostos que lo que sostiene la "
            f"simulación (publicado, mínimo medido): {flojos}"
        )
