using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using MLNet.AutoPipeline;
using MLNet.AutoPipeline.Experiment;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;

namespace MLNet.Expert
{
    public class PipelineBuilder
    {
        public static EstimatorNodeChain BuildPipelineFromState(MLContext context, State state)
        {
            var pipeline = new EstimatorNodeChain();
            pipeline.Append(state.Transformers.SweepablePipelineNodes.Select(x => new EstimatorSingleNode(x)));

            var concatFeaturesTransformer = PipelineBuilder.BuildConcatFeaturesTransformer(context, state.Columns, state.InputOutputColumnPairs);
            pipeline.Append(new EstimatorSingleNode(concatFeaturesTransformer));

            pipeline.Append(state.Trainers);
            return pipeline;
        }

        private static ISweepablePipelineNode BuildConcatFeaturesTransformer(MLContext context, DataViewSchema.Column[] columns, InputOutputColumnPair[] inputOutputColumnPairs, string featureColumnName = "Features") 
        {
            Contract.Requires(columns != null && inputOutputColumnPairs != null);
            var inputColumnNames = inputOutputColumnPairs.Select(x => x.InputColumnName);
            var columnNames = columns.Select(c => c.Name);
            foreach (var name in inputColumnNames)
            {
                Contract.Requires(columnNames.Contains(name), $"input column {name} not exist");
            }

            var outputColumnNames = inputOutputColumnPairs.Select(x => x.OutputColumnName);
            var concatFeaturesTransformer = context.Transforms.Concatenate(featureColumnName, outputColumnNames.ToArray());
            return Util.CreateUnSweepableNode(concatFeaturesTransformer);
        }

        public class State
        {
            public ISweepablePipeline Transformers { get; private set; }

            public DataViewSchema.Column[] Columns { get; private set; }

            public InputOutputColumnPair[] InputOutputColumnPairs { get; private set; }

            public EstimatorNodeGroup Trainers { get; private set; }
        }
    }
}
