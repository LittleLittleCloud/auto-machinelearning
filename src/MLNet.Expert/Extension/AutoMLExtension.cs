// <copyright file="AutoMLExtension.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using MLNet.AutoPipeline;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.Expert.Extension
{
    internal static class AutoMLExtension
    {
        private static Dictionary<AutoPipelineCatalog, SerializableEstimatorCatalog> cache = new Dictionary<AutoPipelineCatalog, SerializableEstimatorCatalog>();

        public static SerializableEstimatorCatalog Serializable(this AutoPipelineCatalog autoPipelineCatalog)
        {
            if (!cache.ContainsKey(autoPipelineCatalog))
            {
                cache.Add(autoPipelineCatalog, new SerializableEstimatorCatalog(autoPipelineCatalog.Context));
            }

            return cache[autoPipelineCatalog];
        }
    }
}
