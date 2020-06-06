# **ML.Net AutoPipeline** : AutoML for ML.NET.
[**ML.Net AutoPipeline**](https://github.com/LittleLittleCloud/machinelearning-auto-pipeline) is a set of libraries build on top of ML.Net that provides AutoML feature. In short, it is aimed to solve the two following problems that vastly exists in Machinelearning:
- Given a MLNet pipeline, find the best hyper-parameters for its transformers or trainers.
- Given a dataset and a ML task, find the best pipeline for solving this task.

**ML.Net AutoPipeline** solves the first problem by sweepable pipeline, which sweeps over a pre-defined hyper-parameter set and finds the best candidate. And it solves the second problem by `MLNet.Expert`, which automatically construct sweepable pipelines based on dataset and tasks. **ML.Net AutoPipeline** contains four libraries:
-  [MLNet.AutoPipeline](https://littlelittlecloud.github.io/machinelearning-auto-pipeline-site/api/MLNet.AutoPipeline.html): Provides API for creating sweepable MLNet pipelines. 
- [MLNet.Sweeper](https://littlelittlecloud.github.io/machinelearning-auto-pipeline-site/api/MLNet.Sweeper.html): Provides different sweepers which can be used to optimize hyper-parameter in `MLNet.AutoPipeline`. Right now the available sweepers are [UniformRandomSweeper](https://littlelittlecloud.github.io/machinelearning-auto-pipeline-site/api/MLNet.Sweeper.UniformRandomSweeper.html), [RandomGridSweeper](https://littlelittlecloud.github.io/machinelearning-auto-pipeline-site/api/MLNet.Sweeper.RandomGridSweeper.html) and [GaussProcessSweeper](https://littlelittlecloud.github.io/machinelearning-auto-pipeline-site/api/MLNet.Sweeper.GaussProcessSweeper.html).
- [MLNet.Expert](https://littlelittlecloud.github.io/machinelearning-auto-pipeline-site/api/MLNet.Expert.html): (coming soon) An AutoML library build on top of `MLNet.AutoPipeline`. It's your best choice if you don't want to define pipeline yourself but want to rely on the power of AutoML. As what it name says, it's your MLNet expert.
- [MLNet.CodeGenerator](https://littlelittlecloud.github.io/machinelearning-auto-pipeline-site/api/MLNet.CodeGenerator.html): (coming soon) Provides API for generating C# code for creating ML.Net pipeline.




# Examples
Please visit our [MLNet-AutoPipeline-Example](https://github.com/LittleLittleCloud/MLNet-AutoPipeline-Examples) for MLNet.AutoPipeline examples.
