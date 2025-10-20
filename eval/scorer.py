'''Adaptaion of the score method for this task from official TEMP implementation'''

def score(prediction_map, ground_truth_map, tax_graph):    
    def get_wu_p(node_a, node_b):
        if node_a not in node2full_path or node_b not in node2full_path:
            return 0.0
        full_path_a = node2full_path[node_a]
        full_path_b = node2full_path[node_b]
        com = full_path_a.intersection(full_path_b)
        
        if not com:
            return 0.0

        lca_dep = 0
        for node in com:
            if node in tax_graph.node2path:
                lca_dep = max(len(tax_graph.node2path[node]), lca_dep)
            
        dep_a = len(tax_graph.node2path.get(node_a, []))
        dep_b = len(tax_graph.node2path.get(node_b, []))
        
        if dep_a + dep_b == 0:
            return 0.0
            
        res = 2.0 * float(lca_dep) / float(dep_a + dep_b)
        return res

    try:
        node2full_path = tax_graph.get_node2full_path()
    except AttributeError:
        '''faced an error with TaxStruct used for training, so had to check this and 
        later switched to the new TaxStruct class from TEMP's implementation for eval'''
        print("Error: Your TaxStruct class needs a 'get_node2full_path' method.") 
        return
        
    wu_p, mrr = 0.0, 0.0
    hit_1, hit_5, hit_10 = 0.0, 0.0, 0.0
    
    n_results = 0 
    for term, predictions in prediction_map.items():
        ground_true = ground_truth_map.get(term)
        
        if not ground_true:
            continue
        if not predictions:
            continue
            
        n_results += 1 

        if predictions[0] == ground_true:
            hit_1 += 1
            
        if ground_true in predictions[:5]:
            hit_5 += 1

        if ground_true in predictions[:10]:
            hit_10 += 1
        rank = 0.0
        for i, r in enumerate(predictions[:1000]): 
            if r == ground_true:
                rank = 1.0 / (i + 1.0)
                break
        mrr += rank
        wu_p += get_wu_p(predictions[0], ground_true)

    if n_results == 0:
        print("Error: No matching terms found between predictions and ground truth.")
        return 0, 0, 0, 0, 0

    n_results_float = float(n_results)
    hit_1 /= n_results_float
    hit_5 /= n_results_float
    hit_10 /= n_results_float
    mrr /= n_results_float
    wu_p /= n_results_float
    
    print(f"Scored {n_results} matching terms.")
    return hit_1, hit_5, hit_10, mrr, wu_p


class EvalArgs:
    def __init__(self):
        self.taxo_path = "science_raw_en.taxo" 
        self.terms_path = "science_eval.terms"
        self.eval_path = "science_eval.gt" 
        self.prediction_file = "/content/merged_predictions.tsv"

if __name__ == "__main__":
    args = EvalArgs()
    ground_truth_map = {}
    query_terms_list = [] 
    try:
        with codecs.open(args.terms_path, 'r', encoding='utf-8') as f_terms, \
             codecs.open(args.eval_path, 'r', encoding='utf-8') as f_parents:
            
            terms = [line.strip() for line in f_terms]
            parents = [line.strip().split("\t")[0] for line in f_parents]
            
            if len(terms) != len(parents):
                print(f"Warning: Mismatch between terms file ({len(terms)}) and parents file ({len(parents)}).")
            
            for term, parent in zip(terms, parents):
                ground_truth_map[term] = parent
                query_terms_list.append(term)
        
        print(f"Loaded {len(ground_truth_map)} ground truth pairs.")
    except FileNotFoundError as e:
        print(f"Error loading ground truth files: {e}")
        exit()

    prediction_map = {}
    try:
        predictions_list = []
        with codecs.open(args.prediction_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split("\t")
                predictions_list.append(parts) 
        print(f"Loaded {len(predictions_list)} prediction lines from {args.prediction_file}")

        if len(query_terms_list) != len(predictions_list):
            print(f"Warning: Mismatch between terms ({len(query_terms_list)}) and predictions ({len(predictions_list)}).")
        
        num_to_score = min(len(query_terms_list), len(predictions_list))
        
        for i in range(num_to_score):
            term = query_terms_list[i]
            predictions = predictions_list[i]
            prediction_map[term] = predictions

        print(f"Created prediction map for {len(prediction_map)} terms based on parallel file order.")

    except FileNotFoundError:
        print(f"Error: Prediction file not found at {args.prediction_file}")
        exit()
        
    try:
        with codecs.open(args.taxo_path, encoding='utf-8') as f:
            tax_lines = f.readlines()
        
        tax_pairs = []
        for line in tax_lines:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                _, child, parent = parts
                tax_pairs.append((parent.strip(), child.strip()))
            elif len(parts) == 2:
                child, parent = parts
                tax_pairs.append((child.strip(), parent.strip()))
            else:
                print(f"Skipping malformed line: {line.strip()}")
        
        tax_graph = TaxStruct(tax_pairs)
        print(f"Loaded taxonomy graph with {len(tax_graph.nodes)} nodes from {args.taxo_path}")
    except FileNotFoundError:
        print(f"Error: file not found at {args.taxo_path}")
        exit()
    h1, h5, h10, mrr, wu_p = score(prediction_map, ground_truth_map, tax_graph)

    print("Finally, the metrics come out to be: ")
    print(f"Hit@1 (Accuracy): {h1 * 100:.2f}%")
    print(f"Hit@5:            {h5 * 100:.2f}%")
    print(f"Hit@10:           {h10 * 100:.2f}%")
    print(f"MRR:              {mrr:.4f}")
    print(f"Wu-Palmer (H@1):  {wu_p:.4f}")
