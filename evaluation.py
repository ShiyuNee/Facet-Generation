import itertools
import pandas as pd
import json
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize
from bert_score.utils import (bert_cos_score_idf, cache_scibert, get_bert_embedding,
                    get_hash, get_idf_dict, get_model, get_tokenizer,
                    lang2model, model2layers, sent_encode)


def best_bleu_cand(groundtruth, candidate):
    assert len(groundtruth) >= len(candidate)
    all_permutations = list(itertools.permutations(candidate))
    max_bleu = 0.
    best_cand = all_permutations[0]
    for cand in all_permutations:
        bleu = 0.
        for i in range(min(len(groundtruth), len(cand))):
            bleu += sentence_bleu([groundtruth[i]], cand[i]) / len(groundtruth)
        if bleu > max_bleu:
            max_bleu = bleu
            best_cand = cand
    return list(best_cand)

def eval_bleu(groundtruth, cand):
    # Calculates the SET BLEU metrics, for 1-gram, 2-gram, 3-gram and 4-gram overlaps
    tokenized_ref = [word_tokenize(t) for t in groundtruth]
    tokenized_cand = [word_tokenize(t) for t in cand]
    best_cand = best_bleu_cand(tokenized_ref, tokenized_cand)
    bleu = [0., 0., 0., 0.]
    bleu_weights = [[1, 0, 0, 0], [0.5, 0.5, 0, 0], [0.33, 0.33, 0.33, 0], [0.25, 0.25, 0.25, 0.25]]
    for j in range(4):
        for i in range(min(len(tokenized_ref), len(best_cand))):
            bleu[j] += sentence_bleu([tokenized_ref[i]], best_cand[i], weights=bleu_weights[j]) / len(tokenized_ref)
    return bleu

def bertscore(groundtruth, cand, tokenizer, model):
    # Calculates the Set BERT-Score metrics for Precision, Recall & F1
    best_cand = best_bleu_cand(groundtruth, cand)
    (P, R, F), hashname = score(best_cand, groundtruth, tokenizer, model, lang="en", return_hash=True, device="cuda:0")
    return float(P.mean().item()), float(R.mean().item()), float(F.mean().item())


def exact_match(groundtruth, cand):
    # Calculates the exact match Precision, Recall & F1
    c = 0.
    for x in cand:
        if x != '' and x in groundtruth:
            c += 1
    p = c / (len([x for x in cand if x != ''])+1e-8)
    r = c / (len([x for x in groundtruth if x != ''])+1e-8)
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
    return [p, r, f1]


def term_match(groundtruth, cand):
    # Calculates the term overlap Precision, Recall & F1
    gt_terms = set([])
    for x in groundtruth:
        if x == '':
            continue
        for t in x.strip().split():
            gt_terms.add(t)
    cand_terms = set([])
    for x in cand:
        if x == '':
            continue
        for t in x.strip().split():
            cand_terms.add(t)

    c = 0.
    for x in cand_terms:
        if x != '' and x in gt_terms:
            c += 1
    p = c / (len([x for x in cand_terms if x != ''])+1e-8)
    r = c / (len([x for x in gt_terms if x != ''])+1e-8)
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
    return [p, r, f1]

def load_tokenizer_model(model_type=None, use_fast_tokenizer=False, num_layers=None, all_layers=False):
    lang = 'en'
    if model_type is None:
        lang = lang.lower()
        model_type = lang2model[lang]
    if num_layers is None:
        num_layers = model2layers[model_type]
    tokenizer = get_tokenizer(model_type, use_fast_tokenizer)
    model = get_model(model_type, num_layers, all_layers)
    return tokenizer, model
# def eval_diversity(terms_set):


if __name__ == "__main__":

    gt_path = MIMICS_MANUAL_PATH
    gt_data = pd.read_csv(gt_path, sep='\t', header=0).to_dict()
    tokenizer, model = load_tokenizer_model()
    gt_query = gt_data['query']

    cand_path = PRED_DATA_PATH
    cand_data = json.loads(open(cand_path).read())

    avg_term_match = [0, 0, 0]
    avg_exact_match = [0, 0, 0]
    avg_bert_score = [0, 0, 0]
    avg_bleu_score = [0, 0, 0, 0]
    cnt = 0

    all_query = []
    for idx in range(len(gt_query)):
        gt_term = []
        cand_term = []
        cnt += 1
        for k in ['option_1', 'option_2', 'option_3', 'option_4', 'option_5']:
            if str(gt_data[k][idx]) != 'nan':
                gt_term.append(gt_data[k][idx])

        cand_term = list(set(cand_data[gt_query[idx]]))
        while len(cand_term) > len(gt_term):
            gt_term.append('')

        while len(gt_term) > len(cand_term):
            cand_term.append('') 

        term_overlap_metrics = term_match(gt_term, cand_term)
        exact_match_metrics = exact_match(gt_term, cand_term)
        bleu_score_metrics = eval_bleu(gt_term, cand_term)
        bert_score_metrics = bertscore(gt_term, cand_term, tokenizer, model)
        
        for idx in range(len(avg_term_match)):
            avg_term_match[idx] = avg_term_match[idx] + term_overlap_metrics[idx]
            avg_exact_match[idx] = avg_exact_match[idx] + exact_match_metrics[idx]
            avg_bert_score[idx] = avg_bert_score[idx] + bert_score_metrics[idx]
            avg_bleu_score[idx] = avg_bleu_score[idx] + bleu_score_metrics[idx]
        avg_bleu_score[3] = avg_bleu_score[3] + bleu_score_metrics[3]



    print("Term overlap metrics: P={},R={},F1={}".format(avg_term_match[0] / cnt,
                                                            avg_term_match[1] / cnt,
                                                            avg_term_match[2] / cnt))

    # exact_match_metrics = exact_match(groundtruth, cand)
    print("Exact match metrics: P={},R={},F1={}".format(avg_exact_match[0] / cnt,
                                                        avg_exact_match[1] / cnt,
                                                        avg_exact_match[2] / cnt))

    print("Bleu score metrics: 1-gram={},2-gram={},3-gram={},4-gram={}".format(avg_bleu_score[0] / cnt,
                avg_bleu_score[1] / cnt,
                avg_bleu_score[2] / cnt,
                avg_bleu_score[3] / cnt))

    print("BERT score metrics: P={},R={},F1={}".format(avg_bert_score[0] / cnt,
                                                       avg_bert_score[1] / cnt,
                                                       avg_bert_score[2] / cnt))
    
    print(cnt)