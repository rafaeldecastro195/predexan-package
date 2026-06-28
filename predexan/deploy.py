import pandas as pd
import numpy as np

class Feature:
    '''
    Classe agnóstica de tecnologia para determinar features, atribuit valores, validar limites, tratar nulos e aplicar normalização.
    '''
    def __init__(self, df, nome_feature, tipagem, valor_min, valor_max, obrigatoriedade=True):
        self.df = df
        self.nome_feature = nome_feature 
        self.valor_min = valor_min
        self.valor_max = valor_max
        self.obrigatoriedade = obrigatoriedade
        self.tipagem = tipagem

    def processar_valor_feature(self, random=True): 
        '''
        Processa e valida o valor fornecido. Retorna apenas o número final tratado ou levanta uma exceção (ValueError).
        '''
        try:
            if random and self.valor_min is not None and self.valor_max is not None:
                return self.tipagem(np.random.uniform(self.valor_min, self.valor_max))
        
        except Exception as e:
            raise ValueError(f"Ocorreu o seguinte erro ao gerar o valor aleatório: {e}")

    def normalizar_feature(self, escalonador, valor_usuario=None, transformar_apenas=True, div_max=False, achatamento=False):
            '''Método matemático para normalizar a feature usando scikit-learn'''
            try: 
                if valor_usuario is not None:
                    dado_para_escalonar = np.array([[valor_usuario]])
                else:
                    dado_para_escalonar = self.df[[self.nome_feature]]
        
                if transformar_apenas:
                    feature_escalonada = escalonador.transform(dado_para_escalonar)
                else:
                    feature_escalonada = escalonador.fit_transform(dado_para_escalonar)
            
                if div_max and feature_escalonada.max() != 0:
                    feature_escalonada = feature_escalonada / feature_escalonada.max()
            
                if achatamento:
                    feature_escalonada = feature_escalonada.flatten()
                    
                if valor_usuario is not None:
                    return float(feature_escalonada[0][0])
                    
                return feature_escalonada
            except Exception as e:
                raise ValueError(f"Ocorreu o seguinte erro ao gerar o valor aleatório: {e}")
            