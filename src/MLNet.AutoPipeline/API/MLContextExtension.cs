// <copyright file="MLContextExtension.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Runtime;

namespace MLNet.AutoPipeline
{
    /// <summary>
    /// Class containing AutoPipeline extension methods to <see cref="MLContext"/>.
    /// </summary>
    public static class MLContextExtension
    {
        private static Dictionary<MLContext, AutoPipelineCatalog> cache = new Dictionary<MLContext, AutoPipelineCatalog>();

        /// <summary>
        /// Extension method for creating <see cref="AutoPipelineCatalog"/>.
        /// </summary>
        /// <param name="context">ML Context.</param>
        /// <returns><see cref="AutoPipelineCatalog"/>.</returns>
        public static AutoPipelineCatalog AutoML(this MLContext context)
        {
            if (!cache.ContainsKey(context))
            {
                Logger.Instance.Channel = (context as IChannelProvider).Start("AutoPipeline");
                cache.Add(context, new AutoPipelineCatalog(context));
            }

            return cache[context];
        }
    }
}
