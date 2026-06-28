import pandas as pd
import numpy as np
import datetime
import time
import tracemalloc
import psutil
import traceback
from sklearn.metrics import precision_recall_curve

def gerar_analise_basica(df, nome_col):
    '''Essa função gera valores de Análise Básica de um ou mais atributos'''
    try:   
        contagem = df[nome_col].count()
        nulas = df[nome_col].isna().sum()
        duplicadas = df[nome_col].size - df[nome_col].nunique()
        valores_unicos = df[nome_col].nunique()
        total_valores = df[nome_col].value_counts()[0]
    
        return contagem, nulas, duplicadas, valores_unicos, total_valores
    
    except ValueError:
        print(f'Erro de valor. Confira a tipagem dos dados enviados. Os dados do atributo {nome_col} são {df[nome_col].dtype}')
        traceback.print_exc()
        return None, None, None, None, None
    except KeyError:
        print(f'A coluna {nome_col} não foi encontrada. Favor digitar uma coluna válida.')
        traceback.print_exc()
        return None, None, None, None, None
    except Exception as e:
        print(f'Ocorreu esse erro: {e}')
        traceback.print_exc()
        return None, None, None, None, None
        
def imprimir_analise_basica(df, nome_col):
    '''Função que mostra os valores das Análises Básicas de um ou mais atributos'''
    try:
        contagem, nulas, duplicadas, valores_unicos, total_valores = gerar_analise_basica(df, nome_col)

        print(f'\n\n Análise Básica do atributo {nome_col}:')
        print(f'\n - No total, há {contagem} células e destas, {nulas} são nulas, {duplicadas} estão repetidas e o total de valores unicos são {valores_unicos}.')

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

def gerar_tendencia_central(df, nome_col):
    '''Função que gera valores de Medidas de Tendência Central de um ou mais atributos'''
    try:
        media = df[nome_col].mean()
        mediana = df[nome_col].median()
        moda = df[nome_col].mode()[0]

        return media, mediana, moda
    
    except ValueError:
        print(f'Erro de valor. Confira a tipagem dos dados enviados. Os dados do atributo {nome_col} são {df[nome_col].dtype}')
        traceback.print_exc()
        return None, None, None
    except KeyError:
        print(f'A coluna {nome_col} não foi encontrada. Favor digitar uma coluna válida.')
        traceback.print_exc()
        return None, None, None
    except Exception as e:
        print(f'Ocorreu esse erro: {e}')
        traceback.print_exc()
        return None, None, None
        
def imprimir_tendencia_central(df, nome_col):
    '''Função que imprime os valores das Medidas de Tendência Central de cada atributo'''
    try:
        media, mediana, moda = gerar_tendencia_central(df, nome_col)

        print(f'\n\n Medidas de Tendência Central do atributo {nome_col}: ')
        print(f'\n - A média aritmética de seus valores é, aproximadamente, {media:.2f};')
        print(f'\n - A mediana é, aproximadamente, {mediana:.2f};')
        print(f'\n - A moda é, aproximadamente, {moda:.2f};')

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

def gerar_dispersao(df, nome_col):
    '''Função que gera valores de  Medidas de Dispersão e Amplitude de um ou mais atributos'''
    try:
        media = df[nome_col].mean()
        valor_min = df[nome_col].min()
        valor_max = df[nome_col].max()
        faixa = valor_max - valor_min
        desvio_padrao = df[nome_col].std()
        coeficiente_variacao = (desvio_padrao/media) if media != 0 else 0

        return media, valor_min, valor_max, desvio_padrao, coeficiente_variacao, faixa

    except ValueError:
        print(f'Erro de valor. Confira a tipagem dos dados enviados. Os dados do atributo {nome_col} são {df[nome_col].dtype}')
        traceback.print_exc()
        return None, None, None, None, None, None
    except KeyError:
        print(f'A coluna {nome_col} não foi encontrada. Favor digitar uma coluna válida.')
        traceback.print_exc()
        return None, None, None, None, None, None
    except Exception as e:
        print(f'Ocorreu esse erro: {e}')
        traceback.print_exc()
        return None, None, None, None, None, None
        
def imprimir_dispersao(df, nome_col):
    '''Imprime Medidas de Dispersão e Amplitude com interpretação estatística'''
    try:
        media, valor_min, valor_max, desvio_padrao, coeficiente_variacao, faixa = gerar_dispersao(df, nome_col)
        
        print('\n\n Medidas de Dispersão e Amplitude')
        print(f'\n - Amplitude: de {valor_min:.2f} até {valor_max:.2f}. Variam em uma faixa de {faixa:.2f}')
        print(f'\n - Desvio Padrão: {desvio_padrao:.2f}')
        print(f'\n - Coeficiente de Variação: {(coeficiente_variacao):.2f} (Valores > 1 indicam alta dispersão)')
        
        if desvio_padrao > media:
             print(f'Nota: O desvio padrão é superior à média, confirmando a presença de outliers e assimetria.')
        else:
             print(f'Os valores concentram-se num intervalo de {media-desvio_padrao:.2f} a {media+desvio_padrao:.2f} (em caso de normalidade).')

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

def gerar_quartis(df, nome_col):
    '''Função que gera valores de  Medidas de Dispersão e Amplitude de um ou mais atributos'''
    try:
        quartil_25 = df[nome_col].quantile(0.25)
        quartil_50 = df[nome_col].quantile(0.50)
        quartil_75 = df[nome_col].quantile(0.75)

        return quartil_25, quartil_50, quartil_75
        
    except ValueError:
        print(f'Erro de valor. Confira a tipagem dos dados enviados. Os dados do atributo {nome_col} são {df[nome_col].dtype}')
        traceback.print_exc()
        return None, None, None
    except KeyError:
        print(f'A coluna {nome_col} não foi encontrada. Favor digitar uma coluna válida.')
        traceback.print_exc()
        return None, None, None
    except Exception as e:
        print(f'Ocorreu esse erro: {e}')
        traceback.print_exc()
        return None, None, None
        
def imprimir_quartis(df, nome_col):
    '''Função que imprime os valores dos quartis de 25%, 50% e 75% de cada atributo'''
    try:
        quartil_25, quartil_50, quartil_75 = gerar_quartis(df, nome_col)

        print(f'\n\n Medidas de Posição (Quartis) do atributo {nome_col}:')
        print(f'\n - O valor do quartil de 25% é {quartil_25:.2f}. Isto significa que 25% dos dados da amostra do atributo {nome_col} são menores ou iguais a esse valor;')
        print(f'\n - O valor do quartil de 50% é {quartil_50:.2f}. Isto significa que 50% dos dados da amostra do atributo {nome_col} são menores ou iguais a esse valor. Equivale à mediana;')
        print(f'\n - O valor do quartil de 75% é {quartil_75:.2f}. Isto significa que 75% dos dados da amostra do atributo {nome_col} são menores ou iguais a esse valor;')

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

def gerar_valor_outliers(df, nome_col):
    '''Função que gera valores de análises de Outliers de um ou mais atributos'''
    try:
        quartil_25 = df[nome_col].quantile(0.25)
        quartil_50 = df[nome_col].quantile(0.50)
        quartil_75 = df[nome_col].quantile(0.75)
        limite_inferior = quartil_25 - (1.5 * (quartil_75 - quartil_25))
        limite_superior = quartil_75 + (1.5 * (quartil_75 - quartil_25))
        outlier_inferior = df[df[nome_col] < limite_inferior].shape[0]
        percentual_outlier_inferior = (outlier_inferior/df[nome_col].shape[0]) * 100
        outlier_superior = df[df[nome_col] > limite_superior].shape[0]
        percentual_outlier_superior = (outlier_superior/df[nome_col].shape[0]) * 100
    
        return quartil_25, quartil_50, quartil_75, limite_inferior, limite_superior, outlier_inferior, percentual_outlier_inferior, outlier_superior, percentual_outlier_superior
     
    except ValueError:
        print(f'Erro de valor. Confira a tipagem dos dados enviados. Os dados do atributo {nome_col} são {df[nome_col].dtype}')
        traceback.print_exc()
        return None, None, None, None, None, None, None, None, None
    except KeyError:
        print(f'A coluna {nome_col} não foi encontrada. Favor digitar uma coluna válida.')
        traceback.print_exc()
        return None, None, None, None, None, None, None, None, None
    except Exception as e:
        print(f'Ocorreu esse erro: {e}')
        traceback.print_exc()
        return None, None, None, None, None, None, None, None, None
    
def imprimir_outliers(df, nome_col):
    '''Função que determina os limites superior e inferior para detecção de Outliers de um ou mais atributos'''
    try:
        quartil_25, quartil_50, quartil_75, limite_inferior, limite_superior, outlier_inferior, percentual_outlier_inferior, outlier_superior, percentual_outlier_superior = gerar_valor_outliers(df, nome_col)

        print(f'\n\n Detecção de Outliers do atributo {nome_col}:')
        print(f'\n - O limite inferior é igual a, aproximadamente, {limite_inferior}. Valores abaixo deste limite podem ser considerados Outliers Inferiores;')
        print(f'\n - O total de outliers de limite inferior é {outlier_inferior:.2f}, o que corresponde a {percentual_outlier_inferior:.2f}% do total de registros;')
        print(f'\n - O limite superior é igual a, aproximadamente, {limite_superior}. Valores acima deste limite podem ser considerados Outliers Superiores;')
        print(f'\n - O total de outliers de limite superior é {outlier_superior:.2f}, o que corresponde a {percentual_outlier_superior:.2f}% do total de registros;')

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

def gerar_regra_ouro_normal(df, nome_col):
    '''Função que determina os valores da regra de ouro da Distribuição Normal de um ou mais atributos'''
    try:
        media = df[nome_col].mean()
        desvio_padrao = df[nome_col].std()

        percentual_68_acima = media + desvio_padrao
        percentual_68_abaixo = media - desvio_padrao

        percentual_95_acima = media + (2 * desvio_padrao)
        percentual_95_abaixo = media - (2 * desvio_padrao)

        percentual_99_acima = media + (3 * desvio_padrao)
        percentual_99_abaixo = media - (3 * desvio_padrao)

        return media, desvio_padrao, percentual_68_acima, percentual_68_abaixo, percentual_95_acima, percentual_95_abaixo, percentual_99_acima, percentual_99_abaixo

    
    except ValueError:
        print(f'Erro de valor. Confira a tipagem dos dados enviados. Os dados do atributo {nome_col} são {df[nome_col].dtype}')
        traceback.print_exc()
        return None, None, None, None, None, None, None, None
    except KeyError:
        print(f'A coluna {nome_col} não foi encontrada. Favor digitar uma coluna válida.')
        traceback.print_exc()
        return None, None, None, None, None, None, None, None
    except Exception as e:
        print(f'Ocorreu esse erro: {e}')
        traceback.print_exc()
        return None, None, None, None, None, None, None, None
        
def imprimir_regra_ouro_normal(df, nome_col):
    '''Função que determina o percentual da amostra a partir da média e desvio padrão da Curva de Distribuição Normal'''
    try:
        media, desvio_padrao, percentual_68_acima, percentual_68_abaixo, percentual_95_acima, percentual_95_abaixo, percentual_99_acima, percentual_99_abaixo = gerar_regra_ouro_normal(df, nome_col) 
        print(f'\n\n Regra de Ouro da Distribuição Normal do atributo {nome_col}:')
        print(f'\n - 68% da amostra está compreendida entre, aproximadamente, {percentual_68_abaixo:.2f} e {percentual_68_acima:.2f};')
        print(f'\n - 95% da amostra está compreendida entre, aproximadamente, {percentual_95_abaixo:.2f} e {percentual_95_acima:.2f};')
        print(f'\n - 99.7% da amostra está compreendida entre, aproximadamente, {percentual_99_abaixo:.2f} e {percentual_99_acima:.2f}.')

    except ValueError:
        print(f'Erro de valor. Confira a tipagem dos dados enviados. Os dados do atributo {nome_col} são {df[nome_col].dtype}')
        return None, None, None, None, None, None, None, None
    except KeyError:
        print(f'A coluna {nome_col} não foi encontrada. Favor digitar uma coluna válida.')
        return None, None, None, None, None, None, None, None
    except Exception as e:
        print(f'Ocorreu esse erro: {e}')
        return None, None, None, None, None, None, None, None

def calcular_percentual(var, total):
    '''Função que calcula o percentual de uma variável em relação ao todo'''
    try:
        percentual = (((var)/(total))*100)
        
        return percentual

    except ValueError:
        print(f'Erro de valor. Confira a tipagem dos dados enviados.')
        traceback.print_exc()
        return None
    except Exception as e:
        print(f'Ocorreu esse erro: {e}')
        traceback.print_exc()
        return None

def medir_tempo(funcao, *args, **kwargs):
    '''Executa a função e mede o tempo real de processamento'''
    try:
        comeco = time.perf_counter()
        resultado = funcao(*args, **kwargs) # Executa o treino/comando aqui  -> *args transforma o parâmetro em listas e tuplas (é possível colocar quantos argumentos forem necessários) e **kwargs, em dicionário
        fim = time.perf_counter()
        
        return (fim - comeco), resultado
    except Exception as e:
        print(f"Erro na medição: {e}")
        traceback.print_exc()
        return None, None

def imprimir_tempo(funcao, *args, **kwargs):
    '''Função que recebe N argumentos e os repassa para a função de medição. Imprime o tempo de processamento'''
    try:
        # Repassamos tudo que recebemos (*args, **kwargs) para medir_tempo
        tempo, resultado = medir_tempo(funcao, *args, **kwargs)

        if tempo is not None:
            print(f"Tempo total de processamento: {tempo:.4f} segundos.")
        else:
            print("Não foi possível medir o tempo.")
            
        return resultado
    except Exception as e:
        print(f"Erro na impressão: {e}")
        traceback.print_exc()
        return None

def registrar_memoria(funcao, *arg, **kwarg):
    '''Função que gera valores de gasto de memória em MB por função'''
    try:
        tracemalloc.start()
        tracemalloc.reset_peak()

        resultado = funcao(*arg, **kwarg)

        atual_b, pico_b = tracemalloc.get_traced_memory()
        
        atual_mb = atual_b / (1024 * 1024)
        pico_mb = pico_b / (1024 * 1024)

        return resultado, atual_b, pico_b
        
    except ValueError:
        print(f'Erro de valor. Confira a tipagem dos dados enviados.')
        traceback.print_exc()
        return None, None, None
    except Exception as e:
        print(f'Ocorreu esse erro: {e}')
        traceback.print_exc()
        return None, None, None

    finally: # Roda esse trecho, independente do que for
        tracemalloc.stop()

def imprimir_memoria(funcao, *args, **kwargs):
    '''Função que imprime o gasto de memória por processamento de funções e métodos em bytes ou megabytes'''
    try:
        
            resultado, atual_b, pico_bb = registrar_memoria(funcao, *args, **kwargs)
            print(f'O total de memória atual em bytes é {atual_b}B e o pico foi de {pico_b}B.')
            print(f'O total de memória atual em megabytes é {(atual_b/(1024*1024))}B e o pico foi de {(pico_b/(1024*1024))}B.')
            return resultado
    
    except ValueError:
        print(f'Erro de valor. Confira a tipagem dos dados enviados.')
        traceback.print_exc()
        return None
    except Exception as e:
        print(f'Ocorreu esse erro: {e}')
        traceback.print_exc()
        return None

def imprimir_desempenho(funcao, *args, **kwargs):
    '''Função que imprime tempo e memória gastos em processamento'''
    try:
        # Captura data/hora exata do início da execução
        agora_inicio = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        

        tracemalloc.start()
        tracemalloc.reset_peak()

        # Chamada inicial para resetar o contador interno de CPU do psutil
        psutil.cpu_percent(interval=None) 
        processo = psutil.Process()
        ram_inicio = processo.memory_info().rss
        
        inicio = time.perf_counter()

        # A função é executada apenas uma vez
        resultado = funcao(*args, **kwargs)
        
        fim = time.perf_counter()
        
        # Captura estado final
        atual_b, pico_b = tracemalloc.get_traced_memory()
        
        # Calcula a CPU gasta especificamente DURANTE a execução da função
        cpu_medida = psutil.cpu_percent(interval=None)
        ram_final = processo.memory_info().rss
        
        tempo_total = fim - inicio

        agora_fim = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        tracemalloc.stop()
        
        print('\nRELATÓRIO DE DESEMPENHO DE PROCESSAMENTO:')  

        print(f'\n  Execução iniciada em: {agora_inicio}')
        
        print(f"\n    1. TEMPO: {tempo_total:.4f}s ({(tempo_total/60):.2f} min)")
        
        print(f"\n    2. MEMÓRIA RAM (Processo Atual):")
        print(f"        - Pico de processamento: {pico_b/(1024**2):.2f} MB")
        print(f"        - Diferença líquida de RAM no processo: {(ram_final - ram_inicio)/(1024**2):.2f} MB")
        print(f"        - Total RAM alocada para o processo: {ram_final/(1024**2):.2f} MB")

        print(f"\n    3. CPU:")
        print(f"        - Uso de CPU do sistema durante a execução: {cpu_medida}%")

        print(f'\n  Execução concluída em: {agora_fim}\n')
        
        return resultado
    
    except ValueError:
        print(f'Erro de valor. Confira a tipagem dos dados enviados.')
        traceback.print_exc()
        return None
    except Exception as e:
        print(f'Ocorreu esse erro: {e}')
        traceback.print_exc()
        return None

    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()

def extrair_prt(y_teste, y_pred_probabilistica):
    '''Extrai os dados de precisão, revocação e limiares'''
    try:
        precision, recall, thresholds = precision_recall_curve(y_teste, y_predicao_probabilistica)

        return precision, recall, thresholds

    except ValueError:
        print(f'Erro de valor. Confira a tipagem dos dados enviados.')
        traceback.print_exc()
        return None, None, None
    except Exception as e:
        print(f'Ocorreu esse erro: {e}')
        traceback.print_exc()
        return None, None, None



def extrair_recall(y_teste, y_pred, meta_recall):
    '''Função que extrai os valores de Recall'''
    try:
        if meta_recall is None:
            raise ValueError("Para usar 'obj_recall=True', você deve definir uma 'meta_recall' (ex: 0.95).")

        precision, recall, thresholds = precision_recall_curve(y_teste, y_pred)
        indices_seguros = np.where(recall >= meta_recall)[0]
            
        if len(indices_seguros) == 0:
            print(f"Não foi possível encontrar nenhum limiar que atinja a meta de {meta_recall*100}% de Recall.")
            return None, None, None
        else:
            melhor_indice = indices_seguros[-1]
            if melhor_indice >= len(thresholds):
                melhor_indice = len(thresholds) - 1
                
            limiar_seguranca = thresholds[melhor_indice]
            precision_resultante = precision[melhor_indice]
            recall_resultante = recall[melhor_indice]

            return melhor_indice, limiar_seguranca, recall_resultante
    except Exception as e:
        print(f'Ocorreu esse erro no Recall: {e}')
        traceback.print_exc()
        return None, None, None

def extrair_precision(y_teste, y_pred, meta_precision):
    '''Função que extrai os valores de Precision'''
    try:
        if meta_precision is None:
            raise ValueError("Para usar 'obj_precision=True', você deve definir uma 'meta_precision' (ex: 0.80).")
            
        precision, _, thresholds = precision_recall_curve(y_teste, y_pred)
        indices_seguros = np.where(precision >= meta_precision)[0]
            
        if len(indices_seguros) == 0:
            print(f"Não foi possível encontrar nenhum limiar que atinja a meta de {meta_precision*100}% de Precision.")
            return None, None, None
        else:
            melhor_indice = indices_seguros[0]
            if melhor_indice >= len(thresholds):
                melhor_indice = len(thresholds) - 1
                
            limiar_seguranca = thresholds[melhor_indice]
            precision_resultante = precision[melhor_indice]

            return melhor_indice, limiar_seguranca, precision_resultante
    except Exception as e:
        print(f'Ocorreu esse erro na Precision: {e}')
        traceback.print_exc()
        return None, None, None

def extrair_f1(y_teste, y_pred):
    '''Função que extrai os valores de F1-Score'''
    try:
        precision, recall, thresholds = precision_recall_curve(y_teste, y_pred)
        precision, recall = precision[:-1], recall[:-1] # Sincroniza tamanhos com o thresholds
                           
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        indice_f1_max = np.argmax(f1_scores)

        return indice_f1_max, thresholds[indice_f1_max], f1_scores[indice_f1_max]
    except Exception as e:
        print(f'Ocorreu esse erro no F1: {e}')
        traceback.print_exc()
        return None, None, None

def extrair_f2(y_teste, y_pred):
    '''Função que extrai os valores de F2-Score'''
    try:
        precision, recall, thresholds = precision_recall_curve(y_teste, y_pred)
        precision, recall = precision[:-1], recall[:-1]
        
        beta = 2
        f2_scores = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-10)
        indice_f2_max = np.argmax(f2_scores)

        return indice_f2_max, thresholds[indice_f2_max], f2_scores[indice_f2_max]
        
    except Exception as e:
        print(f'Ocorreu esse erro no F2: {e}')
        traceback.print_exc()
        return None, None, None


def imprimir_prt(y_teste, y_pred, obj_recall=True, meta_recall=0.95, obj_precision=False, meta_precision=None, obj_f1=False, obj_f2=False):
    '''Consome os dados das funções especialistas e monta o relatório visual na tela'''
    try:
        precision_base, recall_base, _ = precision_recall_curve(y_teste, y_pred)

        # 1. OBJETIVO: RECALL
        if obj_recall:    
            melhor_indice, limiar, recall_res = extrair_recall(y_teste, y_pred, meta_recall)
            if limiar is not None:
                print(f"\n - Objetivo focado em Recall (meta = {meta_recall*100}%)")
                print(f"      Limiar ideal a ser usado: {limiar:.4f}")
                print(f"      Recall real obtido: {recall_res*100:.2f}% (Ataques detectados)")
                print(f"      Precision real obtida: {precision_base[melhor_indice]*100:.2f}% (Acerto dos alertas disparados)")

        # 2. OBJETIVO: PRECISION
        if obj_precision:    
            melhor_indice, limiar, precision_res = extrair_precision(y_teste, y_pred, meta_precision)
            if limiar is not None:
                print(f"\n - Objetivo focado em Precision (meta >= {meta_precision*100}%)")
                print(f"      Limiar ideal a ser usado: {limiar:.4f}")
                print(f"      Recall real obtido: {recall_base[melhor_indice]*100:.2f}% (Ataques detectados)")
                print(f"      Precision real obtida: {precision_res*100:.2f}% (Acerto dos alertas disparados)")

        # 3. OBJETIVO: F1-SCORE TRADICIONAL
        if obj_f1:
            melhor_indice, limiar, _ = extrair_f1(y_teste, y_pred)
            if limiar is not None:
                print(f"\n - Equilíbrio Simétrico (Max F1-Score)")
                print(f"      Limiar ideal: {limiar:.4f}")
                print(f"      Recall neste ponto: {recall_base[melhor_indice]*100:.2f}% (Ataques detectados)")
                print(f"      Precision neste ponto: {precision_base[melhor_indice]*100:.2f}% (Acerto dos alertas)")

        # 4. OBJETIVO: F2-SCORE
        if obj_f2:
            melhor_indice, limiar, _ = extrair_f2(y_teste, y_pred)
            if limiar is not None:
                print(f"\n - Equilíbrio Otimizado para Recall (Max F2-Score)")
                print(f"      Limiar ideal: {limiar:.4f}")
                print(f"      Recall neste ponto: {recall_base[melhor_indice]*100:.2f}% (Ataques detectados)")
                print(f"      Precision neste ponto: {precision_base[melhor_indice]*100:.2f}% (Acerto dos alertas)")

    except Exception as e:
        traceback.print_exc()
        print(f'Erro ao formatar a impressão dos relatórios: {e}')
        return None
        
def extracao_desempenho_modelo(matriz_confusao):
    '''Função que extrai valores de desempenho do modelo para as classes 0 e 1, a partir da Matriz de Confusão'''
    try:
        (tn, fp), (fn, tp) = matriz_confusao
        total = (tn + fn + tp + fp)

        if total == 0:
            return {
                'total_amostras': 0, 'tn': tn, 'fn': fn, 'tp': tp, 'fp': fp,
                'accuracy': 0,
                'classe_1': {'precision': 0, 'recall': 0, 'f1_score': 0},
                'classe_0': {'precision': 0, 'recall': 0, 'f1_score': 0}
            }

        accuracy = (tp + tn) / total

        precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0

        precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0

        return {
            'total_amostras': total,
            'tn': tn,
            'fn': fn,
            'tp': tp,
            'fp': fp,
            'accuracy': round(accuracy, 4),
            'classe_1': {
                'precision': round(precision_1, 4),
                'recall': round(recall_1, 4),
                'f1_score': round(f1_1, 4)
            },
            'classe_0': {
                'precision': round(precision_0, 4),
                'recall': round(recall_0, 4),
                'f1_score': round(f1_0, 4)
            }
        }

    except ValueError:
        print("Erro de validação: Certifique-se de que a matriz é estritamente 2x2.")
        traceback.print_exc()
        return None
    except TypeError:
        print("Erro de tipo: A entrada precisa ser uma matriz (lista de listas ou array numpy).")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"Ocorreu esse erro: {e}")
        traceback.print_exc()
        return None

def imprimir_desempenho_modelo(matriz_confusao):
    '''Função que imprime as métricas de desempenho do modelo para as classes 0 e 1'''
    try:
        metricas = extracao_desempenho_modelo(matriz_confusao)

        if metricas is None:
            return None

        tn = metricas['tn']
        tp = metricas['tp']
        fn = metricas['fn']
        fp = metricas['fp']
        
        total = metricas.get('total_amostras', tn + tp + fn + fp) 

        acuracia = metricas['accuracy']
        
        
        precision_1 = metricas['classe_1']['precision']
        recall_1 = metricas['classe_1']['recall']
        f1_score_1 = metricas['classe_1']['f1_score']
        
        precision_0 = metricas['classe_0']['precision']
        recall_0 = metricas['classe_0']['recall']
        f1_score_0 = metricas['classe_0']['f1_score']
        
        print(f'O total de predições de teste foi {total}.\n')
        print(f'\n -> Classe 0:')
        print(f'    - Verdadeiro Negativo (TN): há {tn} casos, o que representa, aproximadamente, {calcular_percentual(tn, total):.2f}% do total de predições de teste.')
        print(f'    - Falso Negativo (FN): há {fn} casos, o que representa, aproximadamente, {calcular_percentual(fn, total):.2f}% do total de predições de teste.')
        print(f'\n -> Classe 1:')
        print(f'   - Verdadeiro Positivo (TP): há {tp} casos, o que representa, aproximadamente, {calcular_percentual(tp, total):.2f}% do total de predições de teste.')
        print(f'   - Falso Positivo (FP): há {fp} casos, o que representa, aproximadamente, {calcular_percentual(fp, total):.2f}% do total de predições de teste.')

        print(f'\n\n -> Resultados Gerais:')
        print(f'    - A Acurácia total do modelo se dá por: (tp + tn) / total. Seu valor é {acuracia:.4f}.')
        print(f'    - A quantidade de vezes que o modelo acertou é expressa por TN + TP e é igual a {(tn + tp)}, de um total de {total}. Corresponde a, aproximadamente, {calcular_percentual((tn+tp), total):.2f}% das predições.')
        print(f'    - A quantidade de vezes que o modelo errou é expressa por FN + FP e é igual a {(fn + fp)}, de um total de {total}. Corresponde a, aproximadamente, {calcular_percentual((fn+fp), total):.2f}% das predições.')

        print(f'\n\n -> Métricas por Classe:')
        print(f'\n     Classe 0:')
        print(f'        - O valor da Precision é {precision_0:.4f}.')
        print(f'        - O valor do Recall é {recall_0:.4f}.')
        print(f'        - O valor do F1-Score é {f1_score_0:.4f}.')

        print(f'\n     Classe 1:')
        print(f'        - O valor da Precisão é {precision_1:.4f}.')
        print(f'        - O valor do Recall é {recall_1:.4f}.')
        print(f'        - O valor do F1-Score é {f1_score_1:.4f}.\n\n')

    except ValueError:
        print("Erro de validação: Certifique-se de que a matriz é estritamente 2x2.")
        traceback.print_exc()
        return None
    except TypeError:
        print("Erro de tipo: A entrada precisa ser uma matriz (lista de listas ou array numpy).")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"Ocorreu esse erro: {e}")
        traceback.print_exc()
        return None