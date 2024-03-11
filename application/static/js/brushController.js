BrushControllerClass = function () {
  const that = this;
  that.bigSize = 4;
  //that.normalSize = 1.5;

  that.showed_idx = [];

  that.scatter_brush_area = undefined;
  that.scatter_brush = undefined;

  that.linecharts = [];

  that.set_line_brush = function (linecharts) {
    that.linecharts = linecharts;
    // Change by kelei: now we only have one linechart.
    //linecharts[0].syncList = [linecharts[1], linecharts[2]];
    //linecharts[1].syncList = [linecharts[0], linecharts[2]];
    //linecharts[2].syncList = [linecharts[0], linecharts[1]];

    for (const lc of that.linecharts) {
      that.init_line_brush(lc);
    }
  };
  // extent === null recover, extent === undefined clean, otherwise set extent
  that.update_line_brush = function (extent) {
    if (extent !== null) that.line_extent = extent;
    if (that.line_extent) { // recover line brush
      for (const info of that.linecharts) {
        info.line_brush_area.call(info.line_brush.move, that.line_extent);
      }
    }
  }
  that.update_scatter_brush = function () {
    that.scatter_brush_area = d3.select('#scatterplot').select('svg');
    d3.select("#scatterplot svg g.lasso").remove();

    const filterVisibleData = (x) => (x.filter(function (d) {
      return that.showed_idx.has(d.t) || d.t >= DataLoader.density_max//(parseFloat(d3.select(this).style('opacity')) > 1e-8);
    }));

    that.scatter_brush = d3.lasso()
      .closePathSelect(true)
      .closePathDistance(200)
      .items(that._getScatterItems())
      .targetArea(that.scatter_brush_area)
      .on('start', function () {
        that.scatter_brush.items().classed('not_possible', true);
        that.selected_idx = new Set()
        for (const info of that.linecharts) {
          info.line_brush_area.call(info.line_brush.move, null);
          window.EffectForLinechart.updateXAxisFocus(info, { showed_idx: that.showed_idx });
        }
        //d3.selectAll('.flag').classed('selected', false);
      })
      .on('draw', function () {
        that.scatter_brush.possibleItems()
          .classed('not_possible', false)
          .classed('possible', true);
        that.scatter_brush.notPossibleItems()
          .classed('not_possible', true)
          .classed('possible', false);
        // that.showed_idx = that.scatter_brush.possibleItems().data().map((x) => x.t);
        // that.highlight();
        that.refresh_scatter(filterVisibleData(that.scatter_brush.possibleItems()).data().map((x) => x.t));
      })
      .on('end', function () {
        that.scatter_brush.items()
          .classed('not_possible', false)
          .classed('possible', false);
        // that.showed_idx = that.scatter_brush.selectedItems().data().map((x) => x.t);
        // that.highlight();
        that.scatter_select(filterVisibleData(that.scatter_brush.selectedItems()).data().map((x) => x.t), true)
        that.reset_brush()
        that.line_extent = null;
      });
    that.scatter_brush_area.call(that.scatter_brush);
    const plot_g = document.querySelector('#scatterplot svg g.plot')
    const lasso_g = document.querySelector('#scatterplot svg g.lasso')
    lasso_g.parentNode.insertBefore(lasso_g, plot_g)
  };
/*
  that.set_showed_idx = function (idx) {
    that.showed_idx = idx;
    that.refresh_scatter(null, true);
    if (update_info) ModelLoader.update_single_chunk_info(that.showed_idx);
  };
*/

  that.reset_brush = function (idx) {
    if (idx) {
      that.scatter_show(idx)
      that.line_extent = undefined;
      // if (that.line_brush_area) that.line_brush_area.call(that.line_brush.move, undefined);
    } else {
      if (that.showed_idx) {
        that.scatter_show(that.showed_idx)
      }
      if (that.line_extent) { // recover line brush
        for (const info of that.linecharts) {
          info.line_brush_area.call(info.line_brush.move, that.line_extent);
        }
      }
      d3.selectAll('.contour').remove();
    }
    that.refresh_scatter()
  };
  that._getScatterItems = function () {
    return (d3.select('#scatterplot').selectAll('.scatter').filter(d => !d.hided));
  };

  that.set_highlight_idx = function (idx) {
    that.refresh_scatter(idx)
    that.refresh_brush()
  };

  that.set_select_idx = function(idx) {
    that.scatter_select(idx)
    that.refresh_scatter()
    that.refresh_brush()
    ModelLoader.update_single_chunk_info([...that.showed_idx]);
  }

  that.init_line_brush = function (info) {
    const _linechart_brush = function (_info, extent) {
      const pred = function (d) {
        return extent[0] < _info.x_scale(d.t) && _info.x_scale(d.t) < extent[1];
      };
      // that.showed_idx = DataLoader.curData.filter(pred).map((x) => x.t);
      // that.highlight();
      that.refresh_scatter(DataLoader.curData.filter(pred).map((x) => x.t))
    };
    const _linechart_brush_end = function (_info, extent) {
      // for (const otherinfo of info.syncList) {
      //   otherinfo.line_brush_area.call(otherinfo.line_brush.move, extent);
      // }
      if (!extent) {
        that.set_select_idx([]);
        return;
      }
      // Adjust extent to select all data
      let allRange = _info.x_scale.range();
      allRange = [allRange[0], allRange[allRange.length-1]];
      if ((extent[1]-extent[0])/(allRange[1]-allRange[0]) > 0.98) {
        extent[0] = allRange[0]-1;
        extent[1] = allRange[1]+1;
      }
      const pred = function (d) {
        return extent[0] < _info.x_scale(d.t) && _info.x_scale(d.t) < extent[1];
      };
      let data = DataLoader.curData.filter(pred).map((x) => x.t)
      BrushController.scatter_show(data)
      that.set_select_idx(data);
    };
    info.line_brush = d3.brushX()
      .extent([
        [0, 0],
        [info.area.width, info.area.height],
      ]);
    info.line_brush.on('start brush', function () {
        const extent = d3.event.selection;
        if (extent) {
          _linechart_brush(info, extent);
          // if (d3.event.sourceEvent.type === 'mousedown' || d3.event.sourceEvent.type === 'mousemove') { // drag
          //   // for (const otherInfo of info.syncList) {
          //   //   that.updateSyncSelectRect(otherInfo, extent);
          //   // }
          // } else if (d3.event.sourceEvent.type === 'end') { // is synced by other brush
          //   info.fake_rect.attr('width', 0);
          // }
        }
      })
      .on('end', () => {
        if (d3.event.sourceEvent === null) {
          return;
        }
        if (d3.event.sourceEvent.type === 'mouseup') { // drag
          _linechart_brush_end(info, d3.event.selection);
          that.line_extent = d3.event.selection;
        }
        else if (d3.event.sourceEvent.type === 'end') { // is synced by other brush
        // info.line_brush_area.style('opacity', 1);
          const extent = d3.event.selection;
          if (!extent) {
            that.scatter_unselect_all()
            that.line_extent = null
          }
        }
      });
    info.line_brush_area = info.vn.root.select('g.brush');
    if (info.line_brush_area.empty()) {
      info.line_brush_area = info.vn.root.append('g')
        .attr('class', 'brush')
        .attr('id', info.id + '_line-brush');
      info.line_brush_area.call(info.line_brush);
    }
    // info.fake_rect = info.vn.root.append('rect').attr('class', 'sync-select-rect').style('opacity', 0.3);
  };

  // that.updateSyncSelectRect = function (info, extent) {
  //   // info.line_brush_area.call(info.line_brush.move, null);
  //   info.line_brush_area.style('opacity', 0);
  //   info.fake_rect
  //     .attr('x', extent[0]).attr('y', 0)
  //     .attr('width', extent[1] - extent[0]).attr('height', info.area.height)
  //     .style('fill', 'grey').style('opacity', '0.3');
  // };

  that.refresh_scatter = function(highlighted) {
    highlighted = new Set(highlighted)
    let idx0 = new Set([...that.showed_idx].concat([...highlighted]))
    const idx1 = that.selected_idx
    
    if (DataLoader.current_view == 'scatter diff') {
      DataLoader.compared_plot
        .select('path')
        .attr('transform', d => idx1.has(d.t) ? 'scale(1.3)' : 'scale(1.0)')
        .style('fill', d => idx1.has(d.t) ? d.color : d.light_color)
        .style('opacity', d => idx1.has(d.t) ? 1 : 0.5)
    }

    that._getScatterItems()
      .style('fill', (d) => idx1.has(d.t) || highlighted.has(d.t) ? d.color : d.light_color)
      .style('opacity', d => {
        let visible = (idx0.has(d.t) || d.t >= DataLoader.density_max) ? 1 : 0
        return visible
      })
      .attr('r', (d) => idx1.has(d.t) || highlighted.has(d.t) ? that.bigSize : d.size);
      for (const info of that.linecharts) {
        EffectForLinechart.updateXAxisFocus(info, { showed_idx: idx0, selected_idx: idx1 });
      }
  };

  that.refresh_brush = function () {
    for (const info of that.linecharts) {
      info.line_brush_area.call(info.line_brush.move, null);
    }
  }

  that.ensure_idx = function() {
      if (!that.selected_idx) that.selected_idx = new Set()
      if (!that.showed_idx) that.showed_idx = new Set()
  }

  that.scatter_select = function(idx) {
    that.ensure_idx()
    for (let x of idx) if (that.showed_idx.has(x) || x >= DataLoader.density_max)
      that.selected_idx.add(x)
    ModelLoader.update_single_chunk_info([...that.selected_idx])
  }

  that.scatter_unselect_all = function(idx) {
    that.ensure_idx()
    that.selected_idx = new Set()
    ModelLoader.update_single_chunk_info([...that.selected_idx])
  }

  that.scatter_unselect = function(idx) {
    that.ensure_idx()
    for (let x of idx) that.selected_idx.delete(x)
    ModelLoader.update_single_chunk_info([...that.selected_idx])
  }

  that.scatter_show = function(idx) {
    that.ensure_idx()
    for (let x of idx) that.showed_idx.add(x)
  }

  that.scatter_hide = function(idx) {
    that.ensure_idx()
    for (let x of idx) {
      that.showed_idx.delete(x)
      that.selected_idx.delete(x)
    }
  }
};