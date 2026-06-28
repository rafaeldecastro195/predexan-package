from .graficos import gerar_curva_normal, gerar_kde, determinar_limite_x, gerar_histograma, gerar_boxplot, gerar_tsne_2d, gerar_scatterplot
from .metricas import gerar_analise_basica, imprimir_analise_basica, gerar_tendencia_central, imprimir_tendencia_central, gerar_dispersao, imprimir_dispersao, gerar_quartis, imprimir_quartis, gerar_valor_outliers, imprimir_outliers, gerar_regra_ouro_normal, imprimir_regra_ouro_normal
from .preproc import drop_outlier, normalizar_feature
from .deploy import Feature

# __all__ = ['gerar_curva_normal', 'gerar_kde', 'determinar_limite_x', 'gerar_histograma', 'gerar_boxplot', 'gerar_tsne_2d', 'gerar_scatterplot', 'gerar_analise_basica', 'imprimir_analise_basica', 'gerar_tendencia_central', 'imprimir_tendencia_central', 'gerar_dispersao', 'imprimir_dispersao', 'gerar_quartis', 'imprimir_quartis', 'gerar_valor_outliers', 'imprimir_outliers', 'gerar_regra_ouro_normal', 'imprimir_regra_ouro_normal', 'drop_outlier', 'normalizar_feature']

__version__ = '0.1.0'