# For contributors

## 1. How to add the task
-  To add tasks to this package, please refer to the [base.py](../base.py)  file and inherit from PiaFactoryBase, PiaModelBase, PiaConfigBase, PiaONNXConfig.


### 1.1 tasks struct
```bash
└─ tasks
    └─SOME_TASK_NAME # ex ) Moment-retrieval, VQA, .....
        ├─models # directory - contains code for the detailed settings of the model you want to use.
        ├─tests #
            .
            .
            .
        ├─base.py # task model base objects ( config, model, onnxconfig)
        └─factory.py # model factory
    └─SOME_TASK_NAME
    └─SOME_TASK_NAME
```

*****

## 2. How to add models
- To add a model to this package, you must inherit and use the ModelBase created in the task.
- When adding a model, please be sure to write pytest code at **packages/tests/test_models/test_ADD_MODEL.py**.

### 2.1 Model struct
```bash
└─ tasks
    └─SOME_TASK_NAME
        ├─models
            └─SOME_MODEL_NAME # ex ) DETR, LLAVA, LLAMA ....
                ├─main.py # Define the model class you want to create in the factory.
                ├─ORIGIN_MODEL_CODE.py #Define the necessary modules for pull from git or defining the model class.
                .
                .
                .
            └─SOME_MODEL_NAME
                ├─main.py
                ├─ORIGIN_MODEL_CODE.py

            └─SOME_MODEL_NAME
                ├─main.py
                ├─ORIGIN_MODEL_CODE.py
```
