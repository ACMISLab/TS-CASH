from tshpo.automl_libs import get_auto_sklearn_classification_search_space

cs = get_auto_sklearn_classification_search_space(y_train=[0, 1])

# print(cs.get_hyperparameter_names())
# print(cs['__choice__'])
prompt = ""
for _model in cs['__choice__'].choices:
    print(_model)
    prompt += f"\nthe hyperparameters of {_model} are: "
    for _hyp_names in cs.get_hyperparameters():
        if _hyp_names.name.startswith(_model):
            _hpy_name = _hyp_names.name.replace(f"{_model}:", "")
            prompt += f"{_hpy_name},"
    prompt = prompt[:-1] + ";"

print(prompt)
