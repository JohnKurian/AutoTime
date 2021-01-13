import os
import pymongo
from bson.json_util import dumps
import json

client = pymongo.MongoClient(
     "mongodb+srv://root:7dc41992@cluster0.vckj9.mongodb.net/main?retryWrites=true&w=majority")


pipeline = [
   {
     '$match': { 'fullDocument.exp_id': "7f8608e2-14c6-4071-864c-9ae5306324fd" }
   } ]


# runs = []
#
#
# for run in client.automl.runs.find({}):
#      run = json.loads(dumps(run))
#      runs.append(json.loads(dumps(run)))
#
# print(runs)
#
# change_stream = client.automl.runs.watch(pipeline)
# for change in change_stream:
#     print(json.loads(dumps(change))['fullDocument'])
#     print('') # for readability only


best_config = None
for run in client.automl.runs.find({'exp_id': { '$eq': '7f8608e2-14c6-4071-864c-9ae5306324fd' } }).sort([("r2", pymongo.DESCENDING)]):
    best_config = run
    break

print(best_config)

