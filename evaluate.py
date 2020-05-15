from detectron2.evaluation import LVISEvaluator


def get_evaluator(cfg):
    dataset_name = cfg.DATASETS.TEST[0]
    output_folder = 'inference'
    evaluator = LVISEvaluator(dataset_name, cfg, True, output_folder)
    return evaluator





