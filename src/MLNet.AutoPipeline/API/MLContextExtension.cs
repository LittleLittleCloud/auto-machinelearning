// <copyright file="MLContextExtension.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;

namespace MLNet.AutoPipeline
{
    /// <summary>
    /// Class containing AutoPipeline extension methods to <see cref="MLContext"/>.
    /// </summary>
    public static class MLContextExtension
    {
        public static AutoPipelineCatalog AutoML(this MLContext context)
        {
            return new AutoPipelineCatalog(context);
        }
    }
}
