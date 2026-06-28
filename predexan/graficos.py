import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from scipy.stats import norm
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, RocCurveDisplay

# Funções relativas à geração de gráficos de histograma, curva de distribuição normal e kde

def gerar_curva_normal(df, nome_col, cor_cn='k', xy_cn=False, legenda=False, largura_linha=2):
    '''Função que gera uma Curva de Distribuição Normal a ser usada, preferencialmente, compondo outros gráficos'''
    try:
        # Determina os valores da Curva de Distribuíção Normal
        mu = np.mean(df[nome_col])
        sigma = np.std(df[nome_col])
        x = np.linspace(df[nome_col].min(), df[nome_col].max(), 100)

        # Cria a Curva de Distribuição Normal
        curva_normal = plt.plot(x, norm.pdf(x, mu, sigma), color=cor_cn, lw=largura_linha, label='Curva de Distribuição Normal')

        # Legenda mostrando a Curva de Distribuição Normal
        if legenda:
            plt.legend(loc='upper right')
        return curva_normal

    except ValueError:
        print(f'Erro de valor. Confira a tipagem dos dados enviados. Os dados do atributo {nome_col} são {df[nome_col].dtype}')
        traceback.print_exc()
        return None
    except KeyError:
        print(f'A coluna {nome_col} não foi encontrada. Favor digitar uma coluna válida.')
        traceback.print_exc()
        return None
    except Exception as e:
        print(f'Ocorreu esse erro: {e}')
        traceback.print_exc()
        return None

def gerar_kde(df, nome_col, cor_kde='#6D0EEB', legenda=False, xy_kde=True):
    '''Função que gera KDE (Estimativa de Densidade de Kernel)'''
    try:
        # Cria o KDE
        kde = sns.kdeplot(data=df[nome_col], color=cor_kde, label='KDE')

        # Legenda mostrando o KDE
        if legenda:
            plt.legend(loc='upper right')
        return kde

    except ValueError:
        print(f'Erro de valor. Confira a tipagem dos dados enviados. Os dados do atributo {nome_col} são {df[nome_col].dtype}')
        traceback.print_exc()
        return None
    except KeyError:
        print(f'A coluna {nome_col} não foi encontrada. Favor digitar uma coluna válida.')
        return None
    except Exception as e:
        print(f'Ocorreu esse erro: {e}')
        traceback.print_exc()
        return None

def determinar_limite_x(df, nome_col, minimo=None, maximo=None, iqr=False):
    '''Função que determina o limite do Eixo X de um histograma, baseado nos valores passados ou Limites Superior e Inferior e IQR'''
    try:
        if iqr:
            q1 = df[nome_col].quantile(0.25)
            q3 = df[nome_col].quantile(0.75)
            limite_iqr = q3 - q1
            minimo = q1 - (1.5 * limite_iqr)
            maximo = q3 + (1.5 * limite_iqr)
            
        if minimo is None:
            minimo = df[nome_col].min()
        if maximo is None:
            maximo = df[nome_col].max()
            
        return plt.xlim(minimo, maximo)
    
    except ValueError:
        print(f'Erro de valor. Confira a tipagem dos dados enviados. Os dados do atributo {nome_col} são {df[nome_col].dtype}')
        traceback.print_exc()
        return None
    except KeyError:
        print(f'A coluna {nome_col} não foi encontrada. Favor digitar uma coluna válida.')
        traceback.print_exc()
        return None
    except Exception as e:
        print(f'Ocorreu esse erro: {e}')
        traceback.print_exc()
        return None

def determinar_limite_y(df, nome_col, minimo=None, maximo=None):
    '''Função que determina o limite do Eixo X de um histograma, baseado nos valores passados'''
    try:
        limite_atual_inferior, limite_atual_superior = plt.ylim()
        
        if minimo is None:
            minimo = limite_atual_inferior
        if maximo is None:
            maximo = limite_atual_superior
                
        return plt.ylim(minimo, maximo)
        
    except ValueError:
        print(f'Erro de valor. Confira a tipagem dos dados enviados. Os dados do atributo {nome_col} são {df[nome_col].dtype}')
        traceback.print_exc()
        return None
    except KeyError:
        print(f'A coluna {nome_col} não foi encontrada. Favor digitar uma coluna válida.')
        traceback.print_exc()
        return None
    except Exception as e:
        print(f'Ocorreu esse erro: {e}')
        traceback.print_exc()
        return None

def gerar_histograma(df, nome_col, largura=12, altura=4, legenda=False, curva_normal=False, kde=False, cor='#E6B75C', stat='density', largura_linha=2, limite_x=False, minx=None, maxx=None, iqr=False, limite_y=False, miny=None, maxy=None):
    '''Função que gera histograma com a possibilidade de gerar Curva de Distribuição Normal e KDE sobrepostos'''
    try:
        # Determina o tamanho da figura
        plt.figure(figsize=(largura, altura))
        # Cria o gráfico de histograma
        grafico = sns.histplot(df[nome_col], color=cor, stat=stat, line_kws={'lw':largura_linha})
            
        if curva_normal:
            gerar_curva_normal(df, nome_col)

        if kde:
            gerar_kde(df, nome_col)

        if legenda:
            plt.legend(loc='upper right')

        if limite_x:
            determinar_limite_x(df, nome_col, minx, maxx, iqr)

        if limite_y:
            determinar_limite_y(df, nome_col, miny, maxy)
            
        plt.title(f'Distribuição do atributo {nome_col}, usando a estatística de {stat}')
        plt.xlabel(nome_col)
        plt.ylabel(stat)
        plt.show()
        plt.close()
        return grafico

    except ValueError:
        print(f'Erro de valor. Confira a tipagem dos dados enviados. Os dados do atributo {nome_col} são {df[nome_col].dtype}')
        traceback.print_exc()
        return None
    except KeyError:
        print(f'A coluna {nome_col} não foi encontrada. Favor digitar uma coluna válida.')
        traceback.print_exc()
        return None
    except Exception as e:
        print(f'Ocorreu esse erro: {e}')
        traceback.print_exc()
        return None

#============================================================================================================================================================================================================================================

# Funções para geração de gráficos boxplot

def gerar_boxplot(df, nome_col, altura=4, largura=12, cor='#8B56FC', showfliers=True):
    '''Essa função gera um gráfico do tipo Boxplot'''
    try:
        plt.figure(figsize=(largura, altura))

        grafico = sns.boxplot(x=df[nome_col], color=cor, legend=False, showfliers=showfliers)

        plt.title(f'Boxplot do atributo {nome_col}')
        plt.xlabel(nome_col)

        plt.show()
        plt.close()

        print('\n\n')

        return grafico

    except ValueError:
        print(f'Erro de valor. Confira a tipagem dos dados enviados. Os dados do atributo {nome_col} são {df[nome_col].dtype}')
        return None

    except KeyError:
        print(f'A coluna {nome_col} não foi encontrada. Favor digitar uma coluna válida.')
        return None

    except Exception as e:
        print(f'Ocorreu o seguinte erro: {e}')
        return None

#============================================================================================================================================================================================================================================

# Funções para geração de gráficos scatterplot, t-SNE, regressão logística
   
def gerar_tsne_2d(df, nome_col, num_componentes=2, taxa_aprendizado='auto', inicio='random', perplexidade=30):
    '''Essa função gera um algoritmo de TSNE '''
    try: 
        # Define o array relacionado à feature
        X = df[[nome_col]] # Coloca-se o nome do atributo em lista, para que seja visto como duas dimensões, o que é necessário para o TSNE
    
        # Cria instância da classe TSNE
        X_incorporado = TSNE(n_components=num_componentes, learning_rate=taxa_aprendizado, init=inicio, perplexity=perplexidade, random_state=42).fit_transform(X)

        return X, X_incorporado
        
    except ValueError:
        print(f'Erro de valor. Confira a tipagem dos dados enviados. Os dados do atributo {nome_col} são {df[nome_col].dtype}')
        traceback.print_exc()
        return None, None
        
    except KeyError:
        print(f'A coluna {nome_col} não foi encontrada. Favor digitar uma coluna válida.')
        traceback.print_exc()
        return None, None

    except Exception as e:
        print(f'Ocorreu o seguinte erro: {e}')
        traceback.print_exc()
        return None, None

    
def gerar_scatterplot(df, nome_col, largura=12, altura=4, x=None, y=None, tsne2d=False, visualizacao_rl=False, alpha=0.5):
    '''Essa função gera um Scatterplot com possibilidade de visualização de TSNE 2D'''
    try: 

        plt.figure(figsize=(largura, altura))

        if not tsne2d:
            x = x if x is not None else df.index
            y = y if y is not None else df[nome_col]
            grafico = sns.scatterplot(data=df, x=x, y=y, hue=df[nome_col], alpha=alpha)
            plt.title(f'Scatterplot do atributo {nome_col}')
        if tsne2d:
            X, X_incorporado = gerar_tsne_2d(df, nome_col)
            grafico = sns.scatterplot(x=X_incorporado[:, 0],
                                     y=X_incorporado[:,1],
                                     hue=df[nome_col],
                                     alpha=alpha)
            plt.title(f'Scatterplot, com t-SNE 2D, do atributo {nome_col}')
    
        plt.show()
        plt.close()
    
        print('\n\n')
    
        return grafico

    except ValueError:
        print(f'Erro de valor. Confira a tipagem dos dados enviados. Os dados do atributo {nome_col} são {df[nome_col].dtype}')
        traceback.print_exc()
        return None
        
    except KeyError:
        print(f'A coluna {nome_col} não foi encontrada. Favor digitar uma coluna válida.')
        traceback.print_exc()
        return None

    except Exception as e:
        print(f'Ocorreu o seguinte erro: {e}')
        traceback.print_exc()
        return None

def gerar_regplot(df, nome_col, eixo_y, regressao_logistica=False, ci=95, n_boot=80, cor='#6C31F7'):
    '''Essa função gera um scatterplot de regressão'''
    try:
        grafico_regressao = sns.regplot(data=df, x=nome_col, y=eixo_y, logistic=regressao_logistica, ci=ci, n_boot=n_boot, color=cor)
        plt.gca().set(xlim=(df[nome_col].min(), df[nome_col].max()))
        
        tipo = 'Logística' if regressao_logistica else 'Linear'
        plt.title(f'Scatterplot de Regressão {tipo}: {nome_col} e {eixo_y}')
        
        plt.show()
        plt.close()
        return grafico_regressao
    
    except ValueError:
        print(f'Erro de valor. Confira a tipagem dos dados enviados. Os dados do atributo {nome_col} são {df[nome_col].dtype}')
        traceback.print_exc()
        return None
        
    except KeyError:
        print(f'A coluna {nome_col} não foi encontrada. Favor digitar uma coluna válida.')
        traceback.print_exc()
        return None

    except Exception as e:
        print(f'Ocorreu o seguinte erro: {e}')
        traceback.print_exc()
        return None

#============================================================================================================================================================================================================================================

# Funções para geração Curva ROC-AUC e Curva Precision-Recall

def gerar_curva_pr(y_teste, y_predicao_probabilistica, desbalanceamento=False):
    '''Função que gera gráfico de precision-recall curve'''
    try:   
        precision, recall, thresholds = precision_recall_curve(y_teste, y_predicao_probabilistica)
        
        # Plota a curva do classificador principal
        plt.plot(recall, precision, label='Classificador')
        
        # Define a linha de base de acordo com o desbalanceamento
        if desbalanceamento:
            proporcao_positivos = sum(y_teste) / len(y_teste)
            plt.axhline(y=proporcao_positivos, color='r', linestyle='--', label=f'Base Aleatória ({proporcao_positivos:.2f})')
        else:
            base_aleatoria = np.linspace(0, 1, len(precision))
            plt.plot(base_aleatoria, base_aleatoria, linestyle='--', label='Base Aleatória', color='red')
            
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Curva Precision-Recall')
        plt.legend()
        plt.show()
        plt.close()

    except ValueError as e:
        print(f'Erro de valor. Confira a tipagem e o formato dos dados enviados. Detalhes: {e}')
        traceback.print_exc()
        return None
        
    except Exception as e:
        print(f'Ocorreu o seguinte erro inesperado: {e}')
        traceback.print_exc()
        return None


def gerar_roc_auc(modelo, X_teste, y_teste):
    '''Função que gera gráfico de Curva ROC-AUC'''
    try:   
        curva = RocCurveDisplay.from_estimator(modelo, X_teste, y_teste)
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.show()
        plt.close()

    except ValueError as e:
        print(f'Erro de valor. Confira a tipagem e o formato dos dados enviados. Detalhes: {e}')
        traceback.print_exc()
        return None
        
    except Exception as e:
        print(f'Ocorreu o seguinte erro inesperado: {e}')
        traceback.print_exc()
        return None

def gerar_curva_aprendizado(modelo):
    '''Função que gera curva de aprendizado'''
    try:
        if hasattr(modelo, 'loss_curve_'):
            plt.figure(figsize=(8, 5))
            plt.plot(modelo.loss_curve_)
            plt.title('Curva de Aprendizado')
            plt.xlabel('Iterações')
            plt.ylabel('Erro (Loss)')
            plt.grid(True, linestyle='--', alpha=0.6) # Linhas de grade
            plt.show()
            plt.close()
        else:
            # Mensagem amigável caso seja passado o XGBoost, Stacking ou outro modelo que não possui o atributo
            print(f"O modelo {type(modelo).__name__} não possui o atributo 'loss_curve_'.")
            return None
    except Exception as e:
        print(f'Ocorreu esse erro: {e}')
        traceback.print_exc()
        return None
