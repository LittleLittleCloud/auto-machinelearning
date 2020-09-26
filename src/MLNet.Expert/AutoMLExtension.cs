// <copyright file="AutoMLExtension.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using MLNet.AutoPipeline;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.Expert
{
    public static class AutoMLExtension
    {
        private static Dictionary<AutoPipelineCatalog, SerializableCatalog> cache = new Dictionary<AutoPipelineCatalog, SerializableCatalog>();

        public static Experiment CreateBinaryClassificationExperiment(this AutoPipelineCatalog autoPipelineCatalog, IEnumerable<Column> columns, Experiment.Option option)
        {
            var pipelineBuilder = new PipelineBuilder(TaskType.BinaryClassification, false, true);
            var sweepablePipeline = pipelineBuilder.BuildPipeline(autoPipelineCatalog.Context, columns);
            return autoPipelineCatalog.Context.AutoML().CreateExperiment(sweepablePipeline, option);
        }

        internal static SerializableCatalog Serializable(this AutoPipelineCatalog autoPipelineCatalog)
        {
            if (!cache.ContainsKey(autoPipelineCatalog))
            {
                cache.Add(autoPipelineCatalog, new SerializableCatalog(autoPipelineCatalog.Context));
            }

            return cache[autoPipelineCatalog];
        }
    }
}
