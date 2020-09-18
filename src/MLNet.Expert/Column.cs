// <copyright file="Column.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Reflection.Emit;
using System.Text;

namespace MLNet.Expert
{
    public enum ColumnType
    {
        /// <summary>
        /// numeric type.
        /// </summary>
        Numeric = 0,

        /// <summary>
        /// string type.
        /// </summary>
        String = 1,

        /// <summary>
        /// catagorcial type.
        /// </summary>
        Catagorical = 2,

        /// <summary>
        /// timestamp type.
        /// </summary>
        DateTime = 3,
    }

    /// <summary>
    /// From Microsft.ML.AutoML.
    /// </summary>
    public enum ColumnPurpose
    {
        Ignore = 0,
        Label = 1,
        NumericFeature = 2,
        CategoricalFeature = 3,
        TextFeature = 4,
        Weight = 5,
        ImagePath = 6,
        SamplingKey = 7,
        UserId = 8,
        ItemId = 9,
        GroupId = 10,
    }

    public class Column
    {
        public Column(string name, ColumnType columnType, ColumnPurpose columnPurpose)
        {
            this.Name = name;
            this.ColumnPurpose = columnPurpose;
            this.ColumnType = columnType;
        }

        public string Name { get; private set; }

        public ColumnType ColumnType { get; private set; }

        public ColumnPurpose ColumnPurpose { get; private set; }
    }
}
