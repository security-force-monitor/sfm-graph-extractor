def pretty_print(line, preds):
    words = line.strip().split()
    lengths = [max(len(w), len(p)) for w, p in zip(words, preds)]
    padded_words = [w + (l - len(w)) * ' ' for w, l in zip(words, lengths)]
    padded_preds = [p + (l - len(p)) * ' ' for p, l in zip(preds, lengths)]
    print('words: {}'.format(' '.join(padded_words)))
    print('preds: {}'.format(' '.join(padded_preds)))

line = "The Chief of Army Staff (COAS), Lt. Gen. Yusuf Tukur Buratai, stated this yesterday when he came to the Maxwell Khobe Cantonment of the 3 Armoured Division Jos, where he commissioned 60 renovated blocks"
preds = "O B-TOR I-TOR I-TOR I-TOR O B-RNK I-RNK B-PER I-PER I-PER O O O O O O O O O O O O O B-ORG I-ORG I-ORG O O O O O O O".split()

pretty_print(line, preds)
