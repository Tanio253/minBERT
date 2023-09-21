import torch
from bert import BertModel
sanity_data = torch.load("./sanity_check.data")
# text_batch = ["hello world", "hello neural network for NLP"]
# tokenizer here
sent_ids = torch.tensor([[  101,  1996,  5304,  2003,  2019, 15180,  1011, 18988, 22944,  1010,
          3561,  2007, 15572,  1998, 16278,  1012,   102,     0,     0,     0,
             0,     0,     0],
        [  101,  8434,  1037, 12090,  7954,  2003,  2019,  2396,  2008,  7545,
          6569,  2000,  2119,  1996,  5660,  1998,  1996, 15736,  1012,   102,
             0,     0,     0],
        [  101, 11131,  3418,  8435,  4895,  3726, 12146,  1996,  7800,  1997,
          2627, 24784,  1998,  2037,  9487, 10106,  1012,   102,     0,     0,
             0,     0,     0],
        [  101,  1996, 16684,  2614,  1997,  5975, 12894,  2114,  1996,  5370,
          2003,  1996,  3819,  4019,  2013,  1996,  8488,  1997,  3679,  2166,
          1012,   102,     0],
        [  101,  4083,  1037,  2047,  2653,  7480,  2039,  1037,  2088,  1997,
          6695,  1998, 11598,  2015,  3451,  4824,  1012,   102,     0,     0,
             0,     0,     0],
        [  101,  4526,  2396,  4473,  2149,  2000,  4671,  2256,  6699,  1998,
         15251,  1999,  4310,  1998,  3376,  3971,  1012,   102,     0,     0,
             0,     0,     0],
        [  101, 12607,  2015,  1999,  2974,  2031,  4329,  3550,  1996,  2126,
          2057,  2444,  1998, 11835,  2007,  1996,  2088,  2105,  2149,  1012,
           102,     0,     0],
        [  101,  7118,  2000,  2367,  3032, 14451,  2015,  2149,  2000,  7578,
          8578,  1010,  7443,  1010,  1998,  3052, 17904, 12793,  1012,   102,
             0,     0,     0],
        [  101,  1996,  5416,  2090,  4286,  1998,  4176,  2003,  1037,  9025,
          2000,  1996, 15398,  1998,  2293,  2008,  6526,  1999,  1996,  2088,
          1012,   102,     0],
        [  101,  3752,  1037, 14408, 17441,  2338, 19003,  2149,  2000,  2367,
         22213,  1998,  4372, 13149,  2229,  2256,  9273,  2007,  3716,  1998,
          9647,  1012,   102]])
att_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
# load our model
bert = BertModel.from_pretrained('bert-base-uncased')
outputs = bert(sent_ids, att_mask)
att_mask = att_mask.unsqueeze(-1)
outputs['last_hidden_state'] = outputs['last_hidden_state'] * att_mask
sanity_data['last_hidden_state'] = sanity_data['last_hidden_state'] * att_mask

for k in ['last_hidden_state', 'pooler_output']:
    assert torch.allclose(outputs[k], sanity_data[k], atol=1e-5, rtol=1e-3)
print("Your BERT implementation is correct!")