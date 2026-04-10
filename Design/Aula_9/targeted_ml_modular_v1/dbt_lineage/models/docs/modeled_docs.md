{% docs modeled_dim_teacher %}
Dimensão modelada de professores usada pelo ML. A granularidade é um professor por linha. Ela parte do cadastro bruto e acrescenta agregados de observação, intensidade histórica e flags de completude/qualidade de registro para sustentar análises de contexto, população e elegibilidade.
{% enddocs %}

{% docs modeled_fct_session_clean %}
Fato modelado de sessões limpas. A granularidade é uma sessão por linha. Ele padroniza timestamps, deriva duração e oferece a base de sinais de entrada e tempo de uso.
{% enddocs %}

{% docs modeled_fct_interaction_clean %}
Fato modelado de interações limpas. A granularidade é uma interação por linha. Ele normaliza eventos de produto, classifica famílias de ação, marca flags semânticas de atividade e tenta reconciliar o identificador de aula com o catálogo educacional.
{% enddocs %}

{% docs modeled_fct_formation_clean %}
Fato modelado de formação. A granularidade é um evento de formação por linha. Ele padroniza timestamps, progresso e status de conclusão, e adiciona flags úteis para leitura analítica.
{% enddocs %}

{% docs modeled_fct_mari_conversation_resolved %}
Fato modelado de conversas da Mari IA. A granularidade é um registro resolvido de conversa por linha. Ele unifica a leitura entre formatos legados e estruturados e marca presença de mensagens do usuário e da IA.
{% enddocs %}

{% docs modeled_fct_mari_help_resolved %}
Fato modelado de feedback de ajuda da Mari IA. A granularidade é um evento de ajuda por linha. Ele reconcilia o usuário, o momento do evento e a indicação de que a resposta ajudou.
{% enddocs %}

{% docs modeled_mart_teacher_cluster_ready %}
Mart modelado pronto para leitura descritiva de clusters. A granularidade é um professor por linha. Ele agrega intensidade, recorrência, composição de ações e alguns atributos de contexto para suportar segmentação exploratória.
{% enddocs %}

{% docs modeled_mart_teacher_persona_ready %}
Mart modelado pronto para leitura descritiva de personas. A granularidade é um professor por linha. Ele combina contexto, histórico de atividade, shares de comportamento e métricas agregadas de uso para análises de perfil.
{% enddocs %}
