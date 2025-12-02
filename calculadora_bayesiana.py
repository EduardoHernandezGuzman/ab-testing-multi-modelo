# calculadora_bayesiana.py
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Estilo para los gráficos
sns.set(style="whitegrid")

class CalculadoraClicksBayesiana:
    def __init__(self, alpha_prior_a=1, beta_prior_a=1, alpha_prior_b=1, beta_prior_b=1):
        self.alpha_a = alpha_prior_a
        self.beta_a = beta_prior_a
        self.alpha_b = alpha_prior_b
        self.beta_b = beta_prior_b
        self.historial = []
        self._guardar_estado("A priori")

    def _guardar_estado(self, dia):
        estado = {
            'dia': dia,
            'alpha_a': self.alpha_a,
            'beta_a': self.beta_a,
            'alpha_b': self.alpha_b,
            'beta_b': self.beta_b
        }
        self.historial.append(estado)

    def actualizar_con_datos(self, clicks_a, visitas_a, clicks_b, visitas_b, dia=None):
        datos_dia = {
            'clicks_a': clicks_a,
            'visitas_a': visitas_a,
            'clicks_b': clicks_b,
            'visitas_b': visitas_b
        }

        with pm.Model() as model:
            tasa_a = pm.Gamma('tasa_clicks_a', alpha=self.alpha_a, beta=self.beta_a)
            tasa_b = pm.Gamma('tasa_clicks_b', alpha=self.alpha_b, beta=self.beta_b)

            pm.Poisson('obs_a', mu=tasa_a * visitas_a, observed=clicks_a)
            pm.Poisson('obs_b', mu=tasa_b * visitas_b, observed=clicks_b)

            pm.Deterministic('diferencia', tasa_b - tasa_a)

            trace = pm.sample(2000, tune=1000, chains=2, cores=1, progressbar=False)

        self.alpha_a += clicks_a
        self.beta_a += visitas_a
        self.alpha_b += clicks_b
        self.beta_b += visitas_b

        self._guardar_estado(dia or f"Día {len(self.historial)}")
        self.historial[-1]["trace"] = trace
        self.historial[-1]["datos"] = datos_dia

        # Cálculo de uplift/downlift
        tasa_a_muestral = trace.posterior['tasa_clicks_a'].values.flatten()
        tasa_b_muestral = trace.posterior['tasa_clicks_b'].values.flatten()
        uplift_muestral = (tasa_b_muestral - tasa_a_muestral) / tasa_a_muestral

        self.historial[-1]["uplift"] = {
            "media": np.mean(uplift_muestral),
            "std": np.std(uplift_muestral),
            "ic_95": np.percentile(uplift_muestral, [2.5, 97.5])
        }

    def _resumen(self, muestras):
        return {
            'Media': np.mean(muestras),
            'Desviación estándar': np.std(muestras),
            'IC 95%': np.percentile(muestras, [2.5, 97.5])
        }

    def detectar_ganador(self, umbral_probabilidad = 0.95, umbral_mejora_minima = 0.01):
        if not self.historial or 'trace' not in self.historial[-1]:
            return {
                "ganador": None,
                "decision": "Continuar prueba",
                "razon": "No hay datos suficientes"
            }

        ultimo_trace = self.historial[-1]['trace']
        diff = ultimo_trace.posterior['diferencia'].values.flatten()

        prob_b_mejor = np.mean(diff > 0)
        prob_a_mejor = np.mean(diff < 0)

        tasa_a = self.alpha_a / self.beta_a
        tasa_b = self.alpha_b / self.beta_b
        mejora_relativa = (tasa_b - tasa_a) / tasa_a

        if prob_b_mejor >= umbral_probabilidad and mejora_relativa >= umbral_mejora_minima:
            return {
                "ganador": "B",
                "decision": "Implementar B",
                "razon": f"B es mejor con {prob_b_mejor:.1%} de probabilidad y {mejora_relativa:.1%} de mejora",
                "probabilidad": prob_b_mejor,
                "mejora_relativa": mejora_relativa
            }
        elif prob_a_mejor >= umbral_probabilidad and mejora_relativa <= -umbral_mejora_minima:
            return {
                "ganador": "A",
                "decision": "Mantener A",
                "razon": f"A es mejor con {prob_a_mejor:.1%} de probabilidad y {abs(mejora_relativa):.1%} de mejora",
                "probabilidad": prob_a_mejor,
                "mejora_relativa": mejora_relativa
            }
        else:
            return {
                "ganador": None,
                "decision": "Continuar prueba",
                "razon": "No hay evidencia suficiente para declarar un ganador",
                "probabilidad_b_mejor": prob_b_mejor,
                "mejora_relativa": mejora_relativa
            }

    def mostrar_historial_completo(self):
        for paso in self.historial:
            print(f"\n🗓️  {paso['dia']}")
            print(f"Parámetros:")
            print(f"  Grupo A: alpha={paso['alpha_a']:.1f}, beta={paso['beta_a']:.1f}")
            print(f"  Grupo B: alpha={paso['alpha_b']:.1f}, beta={paso['beta_b']:.1f}")

            if "datos" in paso:
                datos = paso["datos"]
                print(f"Datos del día:")
                print(f"  Grupo A: {datos['clicks_a']} clicks en {datos['visitas_a']} visitas (tasa: {datos['clicks_a']/datos['visitas_a']:.4f})")
                print(f"  Grupo B: {datos['clicks_b']} clicks en {datos['visitas_b']} visitas (tasa: {datos['clicks_b']/datos['visitas_b']:.4f})")

            mean_a = paso['alpha_a'] / paso['beta_a']
            std_a = np.sqrt(paso['alpha_a'] / (paso['beta_a']**2))
            ic_a = np.percentile(np.random.gamma(paso['alpha_a'], 1/paso['beta_a'], 10000), [2.5, 97.5])
            print("Grupo A:")
            print(f"  Media esperada: {mean_a:.4f}")
            print(f"  Desviación estándar: {std_a:.4f}")
            print(f"  IC 95%: [{ic_a[0]:.4f}, {ic_a[1]:.4f}]")

            mean_b = paso['alpha_b'] / paso['beta_b']
            std_b = np.sqrt(paso['alpha_b'] / (paso['beta_b']**2))
            ic_b = np.percentile(np.random.gamma(paso['alpha_b'], 1/paso['beta_b'], 10000), [2.5, 97.5])
            print("Grupo B:")
            print(f"  Media esperada: {mean_b:.4f}")
            print(f"  Desviación estándar: {std_b:.4f}")
            print(f"  IC 95%: [{ic_b[0]:.4f}, {ic_b[1]:.4f}]")

            if "trace" in paso:
                diff = paso['trace'].posterior['diferencia'].values.flatten()
                resumen_diff = self._resumen(diff)
                print("Diferencia (B - A):")
                print(f"  Media: {resumen_diff['Media']:.4f}")
                print(f"  Desviación estándar: {resumen_diff['Desviación estándar']:.4f}")
                print(f"  IC 95%: [{resumen_diff['IC 95%'][0]:.4f}, {resumen_diff['IC 95%'][1]:.4f}]")
                print(f"  Probabilidad de que B > A: {np.mean(diff > 0):.2%}")

                if "uplift" in paso:
                    uplift = paso["uplift"]
                    print("Uplift (relativo B vs A):")
                    print(f"  Media: {uplift['media']:.2%}")
                    print(f"  Desviación estándar: {uplift['std']:.2%}")
                    print(f"  IC 95%: [{uplift['ic_95'][0]:.2%}, {uplift['ic_95'][1]:.2%}]")

                tasa_a_samples = paso['trace'].posterior['tasa_clicks_a'].values.flatten()
                tasa_b_samples = paso['trace'].posterior['tasa_clicks_b'].values.flatten()

                plt.figure(figsize=(10, 5))
                sns.kdeplot(tasa_a_samples, label="Grupo A", fill=True)
                sns.kdeplot(tasa_b_samples, label="Grupo B", fill=True)
                plt.title(f"{paso['dia']} - Distribuciones posteriores")
                plt.xlabel("Tasa de clicks por visita")
                plt.legend()
                plt.show()

                plt.figure(figsize=(10, 4))
                sns.kdeplot(diff, label="Diferencia (B - A)", color="purple", fill=True)
                plt.axvline(0, color="black", linestyle="--")
                plt.title(f"{paso['dia']} - Diferencia de tasa de clicks")
                plt.xlabel("Diferencia en clicks por visita")
                plt.legend()
                plt.show()



