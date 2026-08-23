"""Pipeline de préparation des données — sans fuite (split d'abord).

À remplir en Phase 4 (§4 du notebook) :
- make_preprocessor(scenario: str) -> ColumnTransformer : une fonction, pas un
  objet figé, pour pouvoir régénérer les 4 scénarios (toutes variables /
  sans duration / sans variables sensibles / socle minimal) sans dupliquer
  le code de traitement.
"""
