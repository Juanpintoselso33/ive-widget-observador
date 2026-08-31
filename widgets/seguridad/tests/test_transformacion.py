"""
Tests de la transformación dato crudo -> dummy.

Existen porque los tests de mutación de la huella daban falsa seguridad:
verificaban que cambiar `ESPEC_CRUDA` cambiara el hash, pero no que cambiara la
transformación. Codex encontró el agujero invirtiendo Orsi y Delgado en la
especificación: el hash cambiaba correctamente y las dummies seguían saliendo
igual, porque `train_model` usaba los códigos escritos a mano. Un
reentrenamiento con un codebook invertido habría publicado los dos candidatos
cambiados, con una huella aparentemente válida.

O sea: un test sobre el hash comprueba que el contrato se declara; sólo un test
sobre las dummies comprueba que el contrato se cumple.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import pytest

from widgets.seguridad import config
from widgets.seguridad import train_model as tm


@pytest.fixture
def base_minima():
    """
    Una fila por cada valor que interesa comprobar. Trae todas las columnas que
    exige la validación, con valores válidos.
    """
    filas = []
    for idbal, edad, sexo, educ, dpto, ideol, vic in [
        (1, 25, "Hombre", 1, 1, 0, "No"),                    # Orsi
        (2, 35, "Mujer", 8, 3, 5, "Sí  sin violencia"),      # Delgado
        (3, 50, "Hombre", 7, 5, 9, "Sí  con violencia"),     # blanco
        (4, 65, "Mujer", 3, 1, 2, "No"),                     # no votó
        (5, 45, "Hombre", 9, 2, 7, "No"),                    # no recuerda
    ]:
        filas.append({
            "edad": edad, "sexo": sexo, "nivel_educativo": educ,
            "dpto_ech": dpto, "w_norm": 1.0,
            "estrato": f"dpto-{dpto}",
            "var_241 | Victima de delito ultimos 12 meses": vic,
            "var_242 | Autoubicacion izquierda-derecha (0-10)": ideol,
            config.PREGUNTA["columna"]: "De acuerdo",
        })
    return pd.DataFrame(filas)


def test_los_tramos_ideologicos_salen_de_la_especificacion(base_minima, monkeypatch):
    """
    El caso que encontró Codex, trasladado a la variable que ahora manda. Antes
    era con los códigos de balotaje: invertir Orsi y Delgado en la
    especificación cambiaba el hash y NO cambiaba las dummies, porque
    train_model usaba códigos escritos a mano. Balotaje ya no está en el
    modelo, pero el agujero se puede reabrir en cualquier variable, así que el
    test se mantiene sobre los tramos ideológicos.
    """
    normal = tm.preparar(base_minima)
    # ideologia 0 -> izquierda extrema (0-1); 5 -> Centro (referencia)
    assert normal.loc[0, "ideol_izq_extrema"] == 1
    assert normal.loc[1, "ideol_izq_extrema"] == 0

    espec = {**config.ESPEC_CRUDA,
             "ideol_tramos": [["izq_extrema", 9, 10], ["izquierda", 2, 3],
                              ["centroizq", 4, 4], ["centro", 5, 5],
                              ["centroderecha", 6, 6], ["derecha", 7, 8],
                              ["der_extrema", 0, 1]]}
    monkeypatch.setattr(config, "ESPEC_CRUDA", espec)
    monkeypatch.setattr(tm, "ESPEC_CRUDA", espec)

    invertido = tm.preparar(base_minima)
    assert invertido.loc[0, "ideol_der_extrema"] == 1, (
        "invertir los extremos en ESPEC_CRUDA no invirtió las dummies: "
        "la especificación no se está consumiendo de verdad"
    )
    assert invertido.loc[0, "ideol_izq_extrema"] == 0


def test_un_valor_de_la_escala_sin_tramo_aborta(base_minima, monkeypatch):
    """
    Un valor que no cae en ningún tramo quedaría con todas las dummies en cero,
    o sea silenciosamente dentro de la referencia. Tiene que abortar.
    """
    espec = {**config.ESPEC_CRUDA,
             "ideol_tramos": [["izq_extrema", 0, 1], ["izquierda", 2, 3],
                              ["centroizq", 4, 4], ["centro", 5, 5],
                              ["centroderecha", 6, 6], ["derecha", 7, 7],
                              ["der_extrema", 9, 10]]}
    monkeypatch.setattr(tm, "ESPEC_CRUDA", espec)
    df = base_minima.copy()
    df.loc[0, "var_242 | Autoubicacion izquierda-derecha (0-10)"] = 8
    with pytest.raises(SystemExit, match="no caen en ningún tramo"):
        tm.preparar(df)


def test_los_cortes_de_edad_salen_de_la_especificacion(base_minima, monkeypatch):
    normal = tm.preparar(base_minima)
    assert normal.loc[0, "edad_30_44"] == 0   # 25 años cae en 18-29

    espec = {**config.ESPEC_CRUDA, "edad_cortes": [17, 20, 44, 59, 120]}
    monkeypatch.setattr(tm, "ESPEC_CRUDA", espec)
    movido = tm.preparar(base_minima)
    assert movido.loc[0, "edad_30_44"] == 1, "mover los cortes no movió el tramo"


def test_el_colapso_educativo_sale_de_la_especificacion(base_minima, monkeypatch):
    normal = tm.preparar(base_minima)
    assert normal.loc[1, "educ_ter_comp"] == 1   # código 8 -> terciaria completa

    espec = {**config.ESPEC_CRUDA,
             "educ_colapso": {k: 1 for k in range(1, 11)}}
    monkeypatch.setattr(tm, "ESPEC_CRUDA", espec)
    monkeypatch.setattr(tm, "EDUC_COLAPSO", espec["educ_colapso"])
    colapsado = tm.preparar(base_minima)
    assert colapsado.loc[1, "educ_ter_comp"] == 0, (
        "colapsar todo a una categoría no cambió las dummies educativas"
    )


def test_las_etiquetas_de_victima_son_de_dominio_cerrado(base_minima):
    """
    Una etiqueta no declarada tiene que abortar, no caer en "No fue víctima".
    """
    df = base_minima.copy()
    df.loc[0, "var_241 | Victima de delito ultimos 12 meses"] = "Sí, violencia desconocida"
    with pytest.raises(SystemExit, match="victimización"):
        tm.preparar(df)


def test_el_codigo_de_montevideo_sale_de_la_especificacion(base_minima, monkeypatch):
    normal = tm.preparar(base_minima)
    assert normal.loc[0, "es_montevideo"] == 1   # dpto 1

    espec = {**config.ESPEC_CRUDA, "dpto_montevideo": 3}
    monkeypatch.setattr(tm, "ESPEC_CRUDA", espec)
    movido = tm.preparar(base_minima)
    assert movido.loc[0, "es_montevideo"] == 0
    assert movido.loc[1, "es_montevideo"] == 1


def test_la_cobertura_no_cuenta_perfiles_no_elegibles():
    """
    Los casos con una dummy oculta activa (no recuerda el voto, no se ubica
    ideológicamente, sin dato de victimización) no corresponden a ningún perfil
    que el lector pueda elegir, así que no deben contarse como "observados".
    Contarlos hacía que la UI afirmara que un perfil aparece en la encuesta
    cuando no hay ni un caso exacto.
    """
    import json
    if not config.MODEL_COEFFICIENTS_PATH.exists():
        pytest.skip("El modelo todavía no fue entrenado")
    with open(config.MODEL_COEFFICIENTS_PATH, encoding="utf-8") as f:
        modelo = json.load(f)
    cob = modelo["cobertura_perfiles"]
    assert cob["observados"] <= cob["posibles"]
    assert cob["con_30_o_mas"] <= cob["observados"]
    # Los valores concretos, no sólo la coherencia: los incorrectos anteriores
    # (597 observados, 5 con 30+) también pasaban las dos comprobaciones de
    # arriba. Si la base cambia hay que actualizar estos números a propósito,
    # que es justamente la idea.
    #
    # Con siete tramos ideológicos simétricos y sin balotaje: 1.008 perfiles
    # posibles, 573 observados y 7 con 30 casos o más. La versión con balotaje
    # daba 566 de 1.296 y sólo 4 con 30+, así que la cobertura relativa mejoró
    # (57% contra 44%) aunque haya más categorías.
    assert cob["posibles"] == 1008
    assert cob["observados"] == 573, (
        "la cobertura cambió: si es por un cambio de base, actualizar el "
        "número; si no, revisar que no se estén contando perfiles no elegibles"
    )
    assert cob["con_30_o_mas"] == 7
