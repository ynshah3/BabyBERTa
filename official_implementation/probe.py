"""
Probe roberta models trained in huggingface
"""
import shutil

from transformers.models.roberta import RobertaForMaskedLM

from babybert import configs
from babybert.params import param2default, Params
from babybert.probing import do_probing
from babybert.utils import load_tokenizer
from babybert.io import save_yaml_file


MAX_STEP = 160_000


params = Params.from_param2val(param2default)


if __name__ == '__main__':

    model_results_folder_name = 'huggingface_official_reference'
    path_model_results = configs.Dirs.probing_results / model_results_folder_name
    if path_model_results.exists():
        shutil.rmtree(path_model_results)

    for path_model in (configs.Dirs.root / 'official_implementation').glob(f'*/checkpoint-{MAX_STEP}'):

        # load model and tokenizer
        roberta = RobertaForMaskedLM.from_pretrained(path_model)
        roberta.cuda(0)
        path_tokenizer_config = configs.Dirs.root / 'data' / 'tokenizers' / f'{params.bbpe}.json'
        tokenizer = load_tokenizer(path_tokenizer_config, params.max_num_tokens_in_sequence)

        step = MAX_STEP
        rep = path_model.parent.name
        save_path = path_model_results / str(rep) / 'saves'

        # save basic model info
        if not (path_model_results / 'param2val.yaml').exists():
            save_yaml_file(path_out=path_model_results / 'param2val.yaml',
                           param2val={'framework': 'huggingface',
                                      'is_official': True,
                                      'is_reference': True,
                                      'is_base': False,
                                      })

        # for each probing task
        for sentences_path in configs.Dirs.probing_sentences.rglob('*.txt'):
            do_probing(save_path,
                       sentences_path,
                       roberta,
                       step,
                       params.include_punctuation,
                       params.score_with_mask,
                       tokenizer=tokenizer,
                       verbose=True,
                       )
