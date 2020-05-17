from detectron2.evaluation import LVISEvaluator, COCOEvaluator


def get_evaluator(cfg, data_type='coco'):
    dataset_name = cfg.DATASETS.TEST[0]
    output_folder = 'inference'
    if data_type == 'lvis':
        evaluator = LVISEvaluator(dataset_name, cfg, True, output_folder)
    if data_type == 'coco':
        evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
    return evaluator





