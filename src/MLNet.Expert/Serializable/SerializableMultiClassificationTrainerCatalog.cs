// <copyright file="SerializableMulticlassificationTrainerCatalog.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.Expert
{
    internal class SerializableMultiClassificationTrainerCatalog
    {
        public SerializableMultiClassificationTrainerCatalog(MLContext context)
        {
            this.Context = context;
        }

        public MLContext Context { get; private set; }
    }
}
