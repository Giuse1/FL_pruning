import copy

import torch
import vgg11_custom

model = vgg11_custom.vgg11(pretrained=False)
in_features = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(in_features, out_features=10, bias=True)

starting_dict = copy.deepcopy(model.state_dict())
nonzero_output = {k: torch.load(f"nonzero_indices/output/{k.replace('.weight', '')}.pt") for k in
                          starting_dict.keys() if "weight" in k}

present_rows = [v.nonzero(as_tuple=True)[0].tolist() for v in nonzero_output.values()]
present_rows.insert(0, [0, 1, 2])

present_rows = present_rows[:11]
print(f"len present rows: {len(present_rows)}")
print([len(e) for e in present_rows])
print(present_rows[1])
