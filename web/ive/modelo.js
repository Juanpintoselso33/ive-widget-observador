/* ===========================================================================
   La inferencia del widget IVE, sin nada de interfaz.

   Va en su propio archivo para que el test de paridad pueda cargarlo en Node
   —sin DOM, sin fetch— y comparar sus números contra los de `model.py` en las
   5.760 combinaciones posibles de perfil. Si esto viviera dentro de widget.js,
   cargarlo fuera del navegador explotaría y el test tendría que copiar la
   fórmula, que es la forma clásica de que un test verifique su propia copia en
   vez del código real.

   PORTE LITERAL de `widgets/ive/model.py`. Las referencias (categoría omitida)
   son: 18-24 años, hombre, primaria o menos, nada religioso, interior, sin
   hijos, 1-2 personas y "otros" en balotaje.
   =========================================================================== */

(function (raiz) {
  "use strict";

  // Umbrales de interpretación, de shared/config.py::PROB_THRESHOLDS. La clave
  // de color que trae el Python se descarta a propósito: la paleta del Figma
  // no codifica "bueno" ni "malo".
  var UMBRALES = [
    [70, "muy probable que apoyes"],
    [55, "probable que apoyes"],
    [45, "dividido/a"],
    [30, "probable que te opongas"],
    [0, "muy probable que te opongas"]
  ];

  function calcularZ(coef, p) {
    var edad_25_34 = p.tramoEdad === 2 ? 1 : 0;
    var edad_35_44 = p.tramoEdad === 3 ? 1 : 0;
    var edad_45_54 = p.tramoEdad === 4 ? 1 : 0;
    var edad_55_plus = p.tramoEdad === 5 ? 1 : 0;

    var educ_secundaria = p.nivelEduc === 2 ? 1 : 0;
    var educ_ter_incomp = p.nivelEduc === 3 ? 1 : 0;
    var educ_ter_comp = p.nivelEduc === 4 ? 1 : 0;

    var relig_poco = p.religiosidad === 2 ? 1 : 0;
    var relig_bastante = p.religiosidad === 3 ? 1 : 0;
    var relig_mucho = p.religiosidad === 4 ? 1 : 0;

    var hogar_3_4 = p.hogar === 2 ? 1 : 0;
    var hogar_5_plus = p.hogar === 3 ? 1 : 0;

    var balotaje_martinez = p.balotaje === "martinez" ? 1 : 0;
    var balotaje_lacalle = p.balotaje === "lacalle" ? 1 : 0;

    var mujer_x_relig_mucho = p.esMujer * relig_mucho;
    var mujer_x_tiene_hijos = p.esMujer * p.tieneHijos;

    var z = coef["intercept"];
    z += coef["edad_25_34"] * edad_25_34;
    z += coef["edad_35_44"] * edad_35_44;
    z += coef["edad_45_54"] * edad_45_54;
    z += coef["edad_55_plus"] * edad_55_plus;
    z += coef["es_mujer"] * p.esMujer;
    z += coef["educ_secundaria"] * educ_secundaria;
    z += coef["educ_ter_incomp"] * educ_ter_incomp;
    z += coef["educ_ter_comp"] * educ_ter_comp;
    z += coef["relig_poco"] * relig_poco;
    z += coef["relig_bastante"] * relig_bastante;
    z += coef["relig_mucho"] * relig_mucho;
    z += coef["es_montevideo"] * p.esMontevideo;
    z += coef["tiene_hijos"] * p.tieneHijos;
    z += coef["hogar_3_4"] * hogar_3_4;
    z += coef["hogar_5_plus"] * hogar_5_plus;
    z += coef["balotaje_martinez"] * balotaje_martinez;
    z += coef["balotaje_lacalle"] * balotaje_lacalle;
    z += coef["mujer_x_relig_mucho"] * mujer_x_relig_mucho;
    z += coef["mujer_x_tiene_hijos"] * mujer_x_tiene_hijos;
    return z;
  }

  function predecir(modelo, perfil) {
    return (1 / (1 + Math.exp(-calcularZ(modelo.coefficients, perfil)))) * 100;
  }

  /**
   * Redondeo al par, como el `round()` de Python.
   *
   * NO es un detalle de purista: `Math.round(76.5)` da 77 y Python da 76. El
   * promedio nacional es exactamente 76,5 y TODAS las diferencias por grupo se
   * calculan contra ese número ya redondeado, así que con el redondeo de
   * JavaScript las quince cifras de la grilla saldrían corridas un punto
   * respecto de la versión publicada.
   */
  function redondear(x) {
    var piso = Math.floor(x);
    // EL EMPATE SE DETECTA EXACTO, sin tolerancia. Con un margen de 1e-9,
    // 76.5000000001 se trataba como empate y daba 76 mientras Python daba 77,
    // y 77.4999999999 daba 78 contra 77. El modelo de hoy no cae ahí —su valor
    // más cercano a un medio es 78.50000524— pero un reentrenamiento sí
    // podría. La resta es exacta en coma flotante, así que la comparación
    // directa es la que replica a Python. Lo marcó Codex.
    if (x - piso !== 0.5) return Math.round(x);
    return piso % 2 === 0 ? piso : piso + 1;
  }

  function textoInterpretacion(prob) {
    for (var i = 0; i < UMBRALES.length; i++) {
      if (prob >= UMBRALES[i][0]) return UMBRALES[i][1];
    }
    return UMBRALES[UMBRALES.length - 1][1];
  }

  // Mapeo UI → código del modelo, igual que widgets/ive/config.py.
  var BALOTAJE_UI_A_CODIGO = {
    "No votó/Blanco": "otros",
    "Orsi (FA)": "martinez",
    "Delgado (Coalición)": "lacalle"
  };

  /**
   * Convierte los ÍNDICES de los ocho desplegables en el perfil que come el
   * modelo.
   *
   * Vive acá, pura y aparte del DOM, porque es donde estaba el agujero: la
   * grilla de 5.760 perfiles del test comparaba Python contra JS usando
   * perfiles YA codificados, así que cambiar un `idx + 1` por `idx`, dar vuelta
   * sexo y región o alterar el orden de las opciones dejaba el test en verde
   * mientras todos los lectores recibían otro perfil. Lo marcó Codex.
   *
   * Los códigos numéricos arrancan en 1, como en el Python; los binarios son
   * el índice tal cual; el balotaje es una cadena.
   */
  function perfilDesdeIndices(rangos, i) {
    return {
      tramoEdad: i.tramoEdad + 1,
      esMujer: i.esMujer,
      nivelEduc: i.nivelEduc + 1,
      religiosidad: i.religiosidad + 1,
      esMontevideo: i.esMontevideo,
      tieneHijos: i.tieneHijos,
      hogar: i.hogar + 1,
      balotaje: BALOTAJE_UI_A_CODIGO[rangos.balotaje.labels[i.balotaje]]
    };
  }

  /**
   * El índice que arranca seleccionado en cada desplegable.
   *
   * `variable_ranges` no es homogéneo: en los campos numéricos `default` es el
   * CÓDIGO (base 1), en los binarios es el índice y en balotaje es la cadena
   * del código. Acá se normaliza a índice, que es lo que entiende un <select>.
   */
  function indicePorDefecto(rango) {
    if (rango.labels && rango.labels.length && BALOTAJE_UI_A_CODIGO[rango.labels[0]] !== undefined) {
      for (var j = 0; j < rango.labels.length; j++) {
        if (BALOTAJE_UI_A_CODIGO[rango.labels[j]] === rango.default) return j;
      }
      return 0;
    }
    if (rango.default === undefined) return 0;
    var k = rango.options.indexOf(rango.default);
    return k < 0 ? 0 : k;
  }

  raiz.ModeloIVE = {
    perfilDesdeIndices: perfilDesdeIndices,
    indicePorDefecto: indicePorDefecto,
    BALOTAJE_UI_A_CODIGO: BALOTAJE_UI_A_CODIGO,
    calcularZ: calcularZ,
    predecir: predecir,
    redondear: redondear,
    textoInterpretacion: textoInterpretacion,
    UMBRALES: UMBRALES
  };
})(typeof module !== "undefined" && module.exports ? module.exports : window);
