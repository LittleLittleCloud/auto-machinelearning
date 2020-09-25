// <copyright file="AutoMLExtension.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using MLNet.AutoPipeline;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.Expert
{
    internal static class AutoMLExtension
    {
        private static Dictionary<AutoPipelineCatalog, SerializableCatalog> cache = new Dictionary<AutoPipelineCatalog, SerializableCatalog>();

        public static SerializableCatalog Serializable(this AutoPipelineCatalog autoPipelineCatalog)
        {
            if (!cache.ContainsKey(autoPipelineCatalog))
            {
                cache.Add(autoPipelineCatalog, new SerializableCatalog(autoPipelineCatalog.Context));
            }

            return cache[autoPipelineCatalog];
        }
    }
}
