import json
import pandas as pd

def collect_snippets(data_path):
    """
        - return a dict{key: query, value: doc snippets} 
        - we ignore the queries with no doc snippets
    """
    docs = open(data_path).readlines()
    doc_snippets = {} # query -> list(doc_snippets)
    cnt = 0
    for doc in docs:
        doc = json.loads(doc)  # ['instrumentation', 'queryContext', 'webPages', 'relatedSearches', 'videos', 'rankingResponse', 'pagination']
        query = doc['queryContext']['originalQuery']
        temp_snippet = []
        try:
            for value in doc['webPages']['value']: # there are some lines with no "key:webPages"
                temp_snippet.append(value['snippet'])
            doc_snippets[query] = temp_snippet
        except:
            pass
    return doc_snippets

def build_mimics_data(data_path, doc_snippets):
    train_data = []
    train_label = []
    option_data = {}
    click_data = pd.read_csv(data_path, sep='\t', header=0).to_dict() 
    query_data = click_data['query']
    # 准备 option_data(dict,query-->option list)
    for idx in range(len(query_data)):
        temp_option = []
        temp_query = query_data[idx]
        for k in ['option_1', 'option_2', 'option_3', 'option_4', 'option_5']:
            option = click_data[k][idx]
            if type(option) == str:
                temp_option.append(option)
        option_data[temp_query] = temp_option # {key: query, value: list of options}

    
    for idx in range(len(query_data)):
        temp_query = query_data[idx]
        if temp_query in doc_snippets:
            query_doc = temp_query
            for snippet in doc_snippets[temp_query]: # 将doc_snippet拼接在query后面
                query_doc = query_doc + ' </s> ' + snippet

        # 一个输入对应多个输出
        temp_label = option_data[temp_query][0]
        if len(option_data[temp_query]) > 1:
            for option in option_data[temp_query][1:]:
                temp_label += ' </s> ' + option
        train_data.append(query_doc)
        train_label.append(temp_label)

    print('some cases of data')
    for idx in range(10):
        print(f'input: {train_data[idx]}')
        print(f'label: {train_label[idx]}')
    
    return train_data, train_label

def predict_mimics():
    snippet_data_path = 'PATH_Snippets'
    doc_snippets = collect_snippets(snippet_data_path) # get docs
    dev_data_path = 'PATH_MIMICS-Manual'
    dev_data = []
    dev_query = []
    click_data = pd.read_csv(dev_data_path, sep='\t', header=0).to_dict() 
    query_data = click_data['query']
    
    for idx in range(len(query_data)):
        temp_query = query_data[idx]
        if temp_query in doc_snippets:
            dev_query.append(temp_query)
            query_doc = temp_query
            for snippet in doc_snippets[temp_query]: # 将doc_snippet拼接在query后面
                query_doc = query_doc + ' </s> ' + snippet
            dev_data.append(query_doc)
    return dev_data, dev_query

def predict_your_data(query: list, docs: list):
    dev_data = []
    dev_query = query
    for idx in range(len(query)):
        temp_query = query[idx]
        query_doc = temp_query
        for snippet in docs[idx]: # 将doc_snippet拼接在query后面
            query_doc = query_doc + ' </s> ' + snippet
        dev_data.append(query_doc)
    return dev_data, dev_query
