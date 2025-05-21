from pylibs.affiliation.metrics import pr_from_events
from pylibs.affiliation.generics import convert_vector_to_events
import pprint

vector_pred = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
vector_gt = [0, 0, 0, 1, 0, 0, 0, 1, 1, 1]

events_pred = convert_vector_to_events(vector_pred)  # [(4, 5), (8, 9)]
events_gt = convert_vector_to_events(vector_gt)  # [(3, 4), (7, 10)]
Trange = (0, len(vector_pred))  # (0, 10)

pprint.pprint(pr_from_events(events_pred, events_gt, Trange))
