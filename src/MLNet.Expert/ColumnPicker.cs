﻿// <copyright file="ColumnPicker.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics.Contracts;
using System.Linq;
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

    public class ColumnPicker
    {
        private DataViewSchema columns;
        private IEnumerable<DataViewSchema.Column> numericColumns;
        private Option option;
        private HashSet<DataViewType> numericDataViewType = new HashSet<DataViewType>()
        {
            NumberDataViewType.Double,
            NumberDataViewType.Int16,
            NumberDataViewType.Int32,
            NumberDataViewType.Int64,
            NumberDataViewType.SByte,
            NumberDataViewType.Single,
            NumberDataViewType.UInt16,
            NumberDataViewType.UInt32,
            NumberDataViewType.UInt64,
            NumberDataViewType.Byte,
        };

        public ColumnPicker(IDataView dataView, Option option)
            : base()
        {
            this.columns = dataView.Schema;
            this.option = option;
            this.NumericColumns = this.columns.Where(x => this.numericDataViewType.Contains(x.Type) && !this.option.IgnoreColumns.Contains(x.Name));
            this.TextColumns = this.columns.Where(x => x.Type == TextDataViewType.Instance && !this.option.IgnoreColumns.Contains(x.Name) && !this.option.CatagoricalColumns.Contains(x.Name));
            this.CatagoricalColumns = this.columns.Where(x => x.Type == TextDataViewType.Instance && this.option.CatagoricalColumns.Contains(x.Name));
        }

        public IEnumerable<DataViewSchema.Column> NumericColumns { get; private set; }

        public IEnumerable<DataViewSchema.Column> TextColumns { get; private set; }

        public IEnumerable<DataViewSchema.Column> CatagoricalColumns { get; private set; }

        public IEnumerable<DataViewSchema.Column> AvailableColumns
        {
            get
            {
                return this.NumericColumns.Concat(this.TextColumns).Concat(this.CatagoricalColumns);
            }
        }

        public IEnumerable<DataViewSchema.Column> SelectColumn(IEnumerable<DataViewSchema.Column> selectedColumn, int beamSearch)
        {
            var selectedColumnNames = selectedColumn.Select(x => x.Name).ToImmutableHashSet();
            var candidates = this.AvailableColumns.Where(x => !selectedColumnNames.Contains(x.Name)).ToArray();
            if (candidates.Count() <= beamSearch)
            {
                return candidates;
            }

            var pickIndex = Enumerable.Range(0, candidates.Count()).ToList();
            pickIndex.Shuffle();
            return pickIndex.GetRange(0, beamSearch).Select(i => candidates[i]);
        }

        public IEnumerable<DataViewSchema.Column> SelectColumnsAsStart(int n)
        {
            Contract.Requires(n <= this.AvailableColumns.Count());
            if (this.option.StartFromNumericFeature)
            {
                return this.NumericColumns;
            }

            var pickIndex = Enumerable.Range(0, this.AvailableColumns.Count()).ToList();
            pickIndex.Shuffle();
            return pickIndex.GetRange(0, n).Select(i => this.AvailableColumns.ToArray()[i]);
        }

        public class Option
        {
            public IEnumerable<string> IgnoreColumns = new List<string>();

            public IEnumerable<string> CatagoricalColumns = new List<string>();

            public string LabelColumn;

            public bool StartFromNumericFeature = true;
        }
    }
}