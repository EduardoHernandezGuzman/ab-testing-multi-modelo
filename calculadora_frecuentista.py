# calculadora_frecuentista.py
import numpy as np
from itertools import combinations
from collections import defaultdict
from scipy.stats import norm  # IMPORTANTE

class ConversionFrecuentistaMultiGrupo:
    def __init__(self):
        # Aquí guardaremos todo lo que luego pintará la interfaz
        self.resultados = {}

    def analizar_datos(self, datos_totales):
        """
        datos_totales: dict del estilo:
        {
            'A': {'visitas': 560, 'conv': 47},
            'B': {'visitas': 580, 'conv': 101},
            'C': {...},
            ...
        }
        """
        grupos = list(datos_totales.keys())
        self.resultados['grupos'] = {}

        z_score = norm.ppf(0.975)

        # 1) Resultados por grupo
        for grupo in grupos:
            visitas = datos_totales[grupo]['visitas']
            conv = datos_totales[grupo]['conv']
            tasa_conversion = conv / visitas if visitas > 0 else 0.0

            se = np.sqrt(
                (tasa_conversion * (1 - tasa_conversion)) / visitas
            ) if visitas > 0 else 0.0

            ci_lower = max(0,  tasa_conversion - z_score * se)
            ci_upper = min(1,  tasa_conversion + z_score * se)

            self.resultados['grupos'][grupo] = {
                'visitas': visitas,
                'conv': conv,
                'tasa_conversion': tasa_conversion,
                'std_error': se,
                'ci': (ci_lower, ci_upper),
            }

        # 2) Comparaciones por parejas (A vs B, A vs C, etc.)
        self.resultados['comparaciones'] = {}
        for g1, g2 in combinations(grupos, 2):
            data1 = self.resultados['grupos'][g1]
            data2 = self.resultados['grupos'][g2]

            p1, n1 = data1['tasa_conversion'], data1['visitas']
            p2, n2 = data2['tasa_conversion'], data2['visitas']

            diff_prop = p1 - p2
            se_diff = np.sqrt(
                (p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2)
            ) if n1 > 0 and n2 > 0 else np.inf

            if se_diff > 0:
                z_diff = diff_prop / se_diff
                # Probabilidad de que g1 sea mejor que g2
                prob_g1_mejor = 1 - norm.cdf(z_diff)
            else:
                z_diff = np.nan
                prob_g1_mejor = 0.5

            uplift_mean = (p1 - p2) / p2 if p2 > 0 else np.inf
            ci_diff_lower = diff_prop - z_score * se_diff
            ci_diff_upper = diff_prop + z_score * se_diff

            self.resultados['comparaciones'][f"{g1}_vs_{g2}"] = {
                'diff_mean': diff_prop,
                'diff_ci': (ci_diff_lower, ci_diff_upper),
                'uplift_mean': uplift_mean,
                'prob_g1_mejor': prob_g1_mejor,
                'ganador': (
                    g1 if prob_g1_mejor >= 0.95
                    else (g2 if (1 - prob_g1_mejor) >= 0.95 else None)
                ),
            }

            # Versión inversa (g2_vs_g1), igual que en el original
            self.resultados['comparaciones'][f"{g2}_vs_{g1}"] = {
                'diff_mean': -diff_prop,
                'diff_ci': (-ci_diff_upper, -ci_diff_lower),
                'uplift_mean': (p2 - p1) / p1 if p1 > 0 else np.inf,
                'prob_g1_mejor': 1 - prob_g1_mejor,
                'ganador': (
                    g2 if (1 - prob_g1_mejor) >= 0.95
                    else (g1 if prob_g1_mejor >= 0.95 else None)
                ),
            }

    def obtener_ganador_global(self):
        """
        Copiado del código original:
        decide el ganador global a partir de las comparaciones.
        """
        from collections import defaultdict as ddict

        if 'comparaciones' not in self.resultados:
            return "No hay comparaciones calculadas."

        victorias = ddict(int)
        grupos_participantes = set()

        for clave, stats in self.resultados['comparaciones'].items():
            if "_vs_" in clave:
                g1, g2 = clave.split("_vs_")
                grupos_participantes.add(g1)
                grupos_participantes.add(g2)

                if stats['ganador']:
                    victorias[stats['ganador']] += 1

        if victorias:
            ganador_global = max(victorias, key=victorias.get)
            return ganador_global

        elif grupos_participantes:
            # Si nadie gana claramente, coge el de mayor tasa
            mejor_tasa = -1
            ganador_global = None
            for grupo in grupos_participantes:
                tasa = self.resultados['grupos'][grupo]['tasa_conversion']
                if tasa > mejor_tasa:
                    mejor_tasa = tasa
                    ganador_global = grupo
            if ganador_global:
                return (
                    "No hay un ganador estadísticamente significativo en todas las "
                    f"comparaciones, pero '{ganador_global}' tiene la tasa de conversión más alta."
                )

        return "No hay un ganador claro entre todos los grupos."
