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
        public static SerializableEstimatorCatalog Serializable(this AutoPipelineCatalog autoPipelineCatalog)
        {
            return new SerializableEstimatorCatalog(autoPipelineCatalog.Context);
        }
    }
}
