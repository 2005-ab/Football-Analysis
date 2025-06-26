from roboflow import Roboflow
rf = Roboflow(api_key="3205MH29k2z3u5Ejc3HU") # THIS API KEY IS REVOKED. PLEASE USE YOUR OWN API KEY
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(1)
dataset = version.download("yolov5")