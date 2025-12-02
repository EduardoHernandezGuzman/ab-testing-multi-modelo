# calculadora_bayesiana_conversiones.py
import numpy as np

class CalculadoraConversionesBayesiana:
    """
    Calculadora bayesiana para conversiones 0/1 (por ejemplo: compra / no compra),
    basada en un modelo Beta-Binomial para dos grupos A y B.

    La interfaz imita a CalculadoraClicksBayesiana para que app.py
    pueda usarla igual: .actualizar_con_datos(), .historial, .detectar_ganador(), etc.
    """

    def __init__(self, alpha_prior_a=1, beta_prior_a=1,
                       alpha_prior_b=1, beta_prior_b=1,
                       num_samples=100_000):
        # Priors Beta para A y B
        self.alpha_a = alpha_prior_a
        self.beta_a = beta_prior_a
        self.alpha_b = alpha_prior_b
        self.beta_b = beta_prior_b

        self.num_samples = num_samples
        self.historial = []  # lista de "pasos" (días)

        # Paso 0: estado “a priori”
        self.historial.append({
            "dia": "A priori",
            "alpha_a": self.alpha_a,
            "beta_a": self.beta_a,
            "alpha_b": self.alpha_b,
            "beta_b": self.beta_b,
        })

    def actualizar_con_datos(self, conv_a, visitas_a, conv_b, visitas_b, dia=None):
        """
        Actualiza los priors con los datos de un día:
        - conv_a / visitas_a: conversiones y visitas del grupo A
        - conv_b / visitas_b: conversiones y visitas del grupo B
        """
        dia = dia or f"Día {len(self.historial)}"

        # Posterior A
        alpha_post_a = self.alpha_a + conv_a
        beta_post_a  = self.beta_a + (visitas_a - conv_a)

        # Posterior B
        alpha_post_b = self.alpha_b + conv_b
        beta_post_b  = self.beta_b + (visitas_b - conv_b)

        # Guardamos como nuevos priors para la siguiente iteración
        self.alpha_a, self.beta_a = alpha_post_a, beta_post_a
        self.alpha_b, self.beta_b = alpha_post_b, beta_post_b

        # Muestreo Beta
        muestras_a = np.random.beta(alpha_post_a, beta_post_a, self.num_samples).astype(float)
        muestras_b = np.random.beta(alpha_post_b, beta_post_b, self.num_samples).astype(float)

        # Estadísticos individuales
        mean_a = muestras_a.mean()
        mean_b = muestras_b.mean()
        ci_a   = np.percentile(muestras_a, [2.5, 97.5])
        ci_b   = np.percentile(muestras_b, [2.5, 97.5])

        # Comparación B vs A
        diff   = muestras_b - muestras_a
        uplift = np.where(muestras_a != 0, (muestras_b - muestras_a) / muestras_a, np.nan)
        prob_b_mejor = np.mean(diff > 0)

        uplift_mean = np.nanmean(uplift)
        uplift_ci   = np.nanpercentile(uplift, [2.5, 97.5])

        paso = {
            "dia": dia,
            "alpha_a": alpha_post_a,
            "beta_a": beta_post_a,
            "alpha_b": alpha_post_b,
            "beta_b": beta_post_b,
            "datos": {
                "conversiones_a": conv_a,
                "visitas_a": visitas_a,
                "conversiones_b": conv_b,
                "visitas_b": visitas_b,
            },
            "posterior": {
                "A": {
                    "media": float(mean_a),
                    "ci": ci_a,
                    "muestras": muestras_a,
                },
                "B": {
                    "media": float(mean_b),
                    "ci": ci_b,
                    "muestras": muestras_b,
                },
            },
            "comparacion": {
                "diff": diff,
                "uplift": uplift,
                "prob_b_mejor": float(prob_b_mejor),
                "uplift_media": float(uplift_mean),
                "uplift_ci": uplift_ci,
            },
        }

        self.historial.append(paso)

    def detectar_ganador(self, umbral_probabilidad=0.95, umbral_mejora_minima=0.01):
        """
        Devuelve un dict con la MISMA estructura que CalculadoraClicksBayesiana.detectar_ganador:
        - ganador: "A", "B" o None
        - decision
        - razon
        - probabilidad (si hay ganador)
        - probabilidad_b_mejor (si no hay ganador)
        - mejora_relativa
        """
        if len(self.historial) < 2:
            return {
                "ganador": None,
                "decision": "Continuar prueba",
                "razon": "No hay datos suficientes para declarar un ganador",
                "probabilidad_b_mejor": None,
                "mejora_relativa": None
            }

        ultimo = self.historial[-1]
        comp = ultimo["comparacion"]
        prob_b_mejor = comp["prob_b_mejor"]
        uplift_media = comp["uplift_media"]

        prob_a_mejor = 1 - prob_b_mejor

        # B gana
        if prob_b_mejor >= umbral_probabilidad and uplift_media >= umbral_mejora_minima:
            return {
                "ganador": "B",
                "decision": "Implementar B",
                "razon": f"B es mejor con {prob_b_mejor:.1%} de probabilidad y {uplift_media:.1%} de mejora",
                "probabilidad": prob_b_mejor,
                "mejora_relativa": uplift_media
            }
        # A gana
        elif prob_a_mejor >= umbral_probabilidad and uplift_media <= -umbral_mejora_minima:
            return {
                "ganador": "A",
                "decision": "Mantener A",
                "razon": f"A es mejor con {prob_a_mejor:.1%} de probabilidad y {abs(uplift_media):.1%} de mejora",
                "probabilidad": prob_a_mejor,
                "mejora_relativa": uplift_media
            }
        # Nadie gana todavía
        else:
            return {
                "ganador": None,
                "decision": "Continuar prueba",
                "razon": "No hay evidencia suficiente para declarar un ganador",
                "probabilidad_b_mejor": prob_b_mejor,
                "mejora_relativa": uplift_media
            }

    def mostrar_historial_completo(self):
        """
        Imprime un resumen parecido al de CalculadoraClicksBayesiana,
        para que app.py pueda capturarlo con redirect_stdout.
        """
        for paso in self.historial:
            dia = paso["dia"]
            print(f"\n🗓️  {dia}")
            print(f"Parámetros Beta actuales:")
            print(f"  Grupo A: alpha={paso['alpha_a']:.1f}, beta={paso['beta_a']:.1f}")
            print(f"  Grupo B: alpha={paso['alpha_b']:.1f}, beta={paso['beta_b']:.1f}")

            if "datos" in paso:
                d = paso["datos"]
                print("Datos del día:")
                print(f"  Grupo A: {d['conversiones_a']} conversiones de {d['visitas_a']} visitas")
                print(f"  Grupo B: {d['conversiones_b']} conversiones de {d['visitas_b']} visitas")

            if "posterior" in paso:
                post_a = paso["posterior"]["A"]
                post_b = paso["posterior"]["B"]
                print("Posterior Grupo A:")
                print(f"  Media esperada: {post_a['media']:.4f}")
                print(f"  IC 95%: [{post_a['ci'][0]:.4f}, {post_a['ci'][1]:.4f}]")
                print("Posterior Grupo B:")
                print(f"  Media esperada: {post_b['media']:.4f}")
                print(f"  IC 95%: [{post_b['ci'][0]:.4f}, {post_b['ci'][1]:.4f}]")

            if "comparacion" in paso:
                comp = paso["comparacion"]
                print("Comparación B vs A:")
                print(f"  Uplift medio: {comp['uplift_media']:.4f}")
                print(f"  IC 95% uplift: [{comp['uplift_ci'][0]:.4f}, {comp['uplift_ci'][1]:.4f}]")
                print(f"  Probabilidad de que B > A: {comp['prob_b_mejor']:.2%}")
