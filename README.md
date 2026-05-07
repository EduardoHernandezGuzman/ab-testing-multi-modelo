# Calculadora CRO 2.0

Aplicación interactiva para análisis de tests A/B usando métodos bayesianos y frecuentistas. Construida con Streamlit.

## Requisitos

- Python 3.13+
- pip

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
streamlit run app.py
```

## Modelos disponibles

- **Beta-Binomial**: para datos de conversión (0/1)
- **Gamma-Poisson**: para datos de clicks/recuentos
- **Frecuentista**: test z para múltiples grupos (en desarrollo)

## Estructura

```
├── app.py                                  # Aplicación principal
├── calculadora_bayesiana.py               # Gamma-Poisson bayesiano
├── calculadora_bayesiana_conversiones.py  # Beta-Binomial bayesiano
├── calculadora_frecuentista.py            # Test z multi-grupo
└── requirements.txt                        # Dependencias
```


