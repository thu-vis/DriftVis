'use strict';
(function (my) {
  const that = {
    description: 'Create line chart on svg.',
  };
  my.EffectForLinechart = that;
  that.label_font_size = 14;
  that.color = (d) => ('#666');
  that.feature_view_width = undefined;
  that.feature_view_height = undefined;

  that.convert_legend_name = (str) => (str.includes('_') ? str.split('_')[0] :
    (str == 'ED' ? 'Overview' : 'Original drift'));
  that.convert_legend_name_in_grid = (str) => (str.includes('_') ? str.split('_')[0] :
    'Overview');


  /**
   * Update linechart without sedimentation.
   *
   * @param {*} visual_setting {
   *     area: {x,y,width, height}, refer to the whole area of the linechart and the box x-axis
   * }
   * @param {*} data
   * @param {*} unique_id
   * @param {*} legend_key_list
   * @param {*} sync_animate_time
   * @returns
   */
  that.drawLinechart = function (visual_setting, data, unique_id,
    legend_key_list, sync_animate_time) {
    const container = visual_setting.container;
    let area = visual_setting.area;
    const yLabel = visual_setting.y_label;
    const xLabel = visual_setting.x_label;
    // color = visual_setting.color,
    const color = that.color;
    let animateAxisTime = visual_setting.animateAxisTime;
    let animatePathTime = (sync_animate_time === undefined) ? 1000 : (sync_animate_time - animateAxisTime);

    if (DataLoader.actionFlags['direct-animation'] == true) {
      animateAxisTime = 500;
      animatePathTime = 500;
    }

    let isInit = false;

    const slc = container.select('#' + unique_id);
    let root = undefined;
    let info = {
      id: unique_id,
    };
    if (slc.empty()) {
      root = container.append('g').datum(info).attr('id', unique_id);
      root.attr('transform', 'translate(' + area.x + ',' + area.y + ')');
      info.area = area;
      info.vn = {};
      info.vn.root = root;
      info.vn.xAxis = root.append('g')
        .attr('class', 'x axis');
      info.vn.yAxis = root.append('g')
        .attr('class', 'y axis');
      info.vn.xAxisLabel = root.append('text').style('font-size', that.label_font_size);
      info.vn.yAxisLabel = root.append('text').style('font-size', that.label_font_size);
      info.vn.timeRegionBorder = root.append('g').classed('time-region-border', true);

      info.legendDic = {};
      isInit = true;
    } else {
      slc.each(function (d) {
        info = d;
      });
      root = info.vn.root;
      area = info.area;
    }
    const height = area.height;
    const width = area.width;
    info.olddata = info.data;
    info.data = data.map((d) => (d));
    info.legend_key_list = legend_key_list;

    const xScaleInfo = that.getSegmentXScale(0, width),
      xScale = xScaleInfo.scale;

    let maxH = 0;
    for (const item of data) {
      for (const key of legend_key_list) {
        maxH = Math.max(maxH, item[key]);
      }
    }
    maxH = (maxH > 0.5) ? 1 : (maxH > 0.25) ? 0.5 : 0.25;
    const yAxisHeight = height - visual_setting.x_box.height;
    const yScale = d3.scaleLinear()
      .domain([0, maxH])
      .range([yAxisHeight, 0]);
    info.old_x_scale = info.x_scale;
    info.x_scale = xScale;
    info.old_y_scale = info.y_scale;
    info.y_scale = yScale;
    if (info.glypher === undefined && visual_setting.has_glyph) {
      info.glypher = SVGUtil.getFormatAssigner(DataLoader.visual_encoding.line_glyph);
    }
    const glypher = info.glypher;

    const myScale = {
      x: xScale,
      y: yScale,
    };
    const oldScale = {
      x: info.old_x_scale,
      y: info.old_y_scale,
    };

    const animateFunc = function (_vn) {
      if (isInit) {
        return _vn;
      } else if (info.olddata === undefined || info.olddata.length <= info.data.length) {
        return (_vn.transition().duration(animateAxisTime));
      } else if (info.olddata.length > info.data.length) {
        return (_vn.transition().delay(animatePathTime).duration(animateAxisTime));
      }
    };

    { // render axes and barchart under x-axis.
      xScaleInfo.axis_func(animateFunc(info.vn.xAxis)
        .attr('transform', 'translate(0,' + height + ')'));
      animateFunc(info.vn.yAxis)
        .call(d3.axisLeft(yScale).ticks(5));

      animateFunc(info.vn.xAxisLabel)
        .attr('font-size', '12px')
        .attr('x', width + 6).attr('y', height + 3).text(xLabel);
      animateFunc(info.vn.yAxisLabel)
        .attr('x', 5).attr('y', 10).text(yLabel);
      that.updateXAxisFlags(info, {
          //x: 0, // calculate by xScale
          y: yAxisHeight,
          //width: width, // calculate by xScale
          height: visual_setting.x_box.height
        },
        xScale, visual_setting.x_box.slice_num);

      const timeRegionAnimate = function (_vn) {
        if (isInit) {
          return _vn;
        } else if (info.olddata === undefined || info.olddata.length <= info.data.length) {
          return (_vn.transition().delay(animateAxisTime - 10).duration(10));
        } else if (info.olddata.length > info.data.length) {
          return (_vn.transition().delay(animatePathTime + animateAxisTime - 10).duration(10));
        }
      };
      that.updateTimeRegionBorder(info, timeRegionAnimate);
    }


    const _updatePathGlyph = utils.updatePathGlyph;
    const _updateLineGlyph = utils.updateLineGlyph;
    const _updatePath = utils.updatePath;

    const _updatePathWithAnimation = function (_vn) {
      if (isInit || info.olddata === undefined) {
        return _updatePath(_vn, info.data, myScale);
      }
      if (info.olddata !== undefined) {
        if (info.olddata.length < info.data.length) {
          const pathTweenInfo = utils.pathTween.getIncreasing(info.olddata, info.data,
            oldScale, myScale);
          _vn.transition().duration(animateAxisTime).attrTween('d', pathTweenInfo.axis)
            .transition().duration(animatePathTime).attrTween('d', pathTweenInfo.curve);
        } else {
          const pathTweenInfo = utils.pathTween.getDecreasing(info.olddata, info.data,
            oldScale, myScale);
          _vn.transition().duration(animatePathTime).attrTween('d', pathTweenInfo.curve)
            .transition().duration(animateAxisTime).attrTween('d', pathTweenInfo.axis);
        }
        return _vn;
      }
    };

    const _renderLinechartLegend = function (_legend_info, _legend_xy, _glyph_setting) {
      const lx = _legend_xy.x;
      const ly = _legend_xy.y;
      const key = _legend_xy.name;

      const _updateLine = function (_vn) {
        return _vn.attr('x1', lx - 15)
          .attr('x2', lx + 15)
          .attr('y1', ly)
          .attr('y2', ly)
          .style('stroke', (_key) => color(_key))
          .style('stroke-width', 2);
      };

      const _updateText = function (_vn) {
        return _vn.attr('x', lx + 20)
          .attr('y', ly)
          .text((_key) => that.convert_legend_name(_key)).style('font-size', '12px')
          .attr('alignment-baseline', 'middle');
      };

      if (_legend_info.line === undefined) {
        _legend_info.line = root.append('line').datum(key);
      }
      if (_legend_info.text === undefined) {
        _legend_info.text = root.append('text').datum(key);
      }
      _legend_info.line = _updateLine(_legend_info.line);
      _legend_info.text = _updateText(_legend_info.text);

      if (_glyph_setting === undefined) {
        return;
      }
      if (_legend_info.glyph.line === undefined) {
        _legend_info.glyph.line = root.append('g').attr('class', 'line-glyph').datum(key);
      }
      _updateLineGlyph(_legend_info.glyph.line, glypher.get(key), _legend_xy);
    };

    const _renderLinechartCurve = function (_legend_info, _legend_xy, _glyph_setting) {
      const _key = _legend_xy.name;
      if (_legend_info.path === undefined) {
        _legend_info.path = _updatePath(root.append('path').attr('class', 'line').datum(_key), info.data, myScale);
      } else {
        _updatePathWithAnimation(_legend_info.path);
      }
      if (_glyph_setting === undefined) {
        return;
      }

      const glyph = glypher.get(_legend_xy.name);

      if (_legend_info.glyph.path !== undefined) {
        const tweenInfo = utils.getGlyphTween(_glyph_setting, info.olddata, info.data, oldScale, myScale);
        let tween_list = [{
          time: animateAxisTime,
          tween_func: tweenInfo.axis,
        }, {
          time: animatePathTime,
          tween_func: tweenInfo.curve,
        }];
        if (info.olddata.length >= info.data.length) {
          tween_list = tween_list.reverse();
        }

        utils.updatePathGlyphWithTweenAnimation(_legend_info.glyph.path, tween_list);
      } else {
        _legend_info.glyph.path = _updatePathGlyph(
          root.append('g').attr('class', 'path-glyph').datum(_key),
          _glyph_setting, info.data, glyph, myScale);
      }
    };

    const _clearLinechartAndLegend = function (_legend_info) {
      for (const kk of ['path', 'line', 'text']) {
        if (_legend_info[kk] === undefined)
          continue;
        _legend_info[kk].remove();
      }
      for (const item_key in _legend_info.glyph) {
        _legend_info.glyph[item_key].remove();
      }
    };

    // paint selected lines
    let maxLegendTextWidth = 0;
    legend_key_list.forEach((key) => {
      maxLegendTextWidth = Math.max(maxLegendTextWidth,
        getTextWidth(that.convert_legend_name(key), "12px Helvetica"));
    });

    let legend_y = 0,
      legendBaseWidth = 22,
      baseOffsetX = legendBaseWidth + maxLegendTextWidth;
    for (const key in info.legendDic) {
      if (info.legendDic[key] === undefined) {
        continue;
      }
      info.legendDic[key].isExist = false;
    }

    legend_key_list.forEach(function (legend_key, idx) {
      legend_y += 15;
      const legend_xy = {
        name: legend_key,
        x: width - baseOffsetX,
        y: legend_y - 15,
      };

      let glyphSetting = undefined;
      if (glypher !== undefined) {
        glyphSetting = {
          name: legend_key,
          shift: idx * (1.0 / legend_key_list.length),
        };
      }

      if (info.legendDic[legend_key] === undefined) {
        info.legendDic[legend_key] = {
          glyph: {},
        };
      }
      _renderLinechartCurve(info.legendDic[legend_key], legend_xy, glyphSetting);
      _renderLinechartLegend(info.legendDic[legend_key], legend_xy, glyphSetting);
      info.legendDic[legend_key].isExist = true;
    });

    // Paint legend bounding box.
    let rect = SVGUtil.ensureSpecificSVGItem(root, 'rect', 'legend-bbox'),
      legendRectPadding = 3;
    SVGUtil.attrD3(rect, {
        x: width - baseOffsetX - 19,
        y: -5 - legendRectPadding,
        width: legendBaseWidth + maxLegendTextWidth + 19,
        height: -5 + legend_y + legendRectPadding * 2
      }).style('fill', 'transparent').style('stroke', 'black')
      .style('stroke-width', (legend_y > 0 ? 1 : 0));

    for (const key in info.legendDic) {
      if (info.legendDic[key] === undefined) {
        continue;
      }
      if (info.legendDic[key].isExist == false) {
        _clearLinechartAndLegend(info.legendDic[key]);
        delete info.legendDic[key];
        info.glypher.remove(key);
      }
    }
    return info;
  };

  /**
   * Update axis of linechart when new data come in.
   */
  that.updateAxisWithNewData = function (info, new_data, animate_time) {
    const height = info.area.height;
    const width = info.area.width;
    info.olddata = info.data;
    info.data = new_data.map((d) => (d));
    //console.log('olddata', JSON.parse(JSON.stringify(info.olddata)))
    //console.log('newdata', JSON.parse(JSON.stringify(info.data)))
    const data = info.data;

    const xScaleInfo = that.getSegmentXScale(0, width, new_data.length),
      xScale = xScaleInfo.scale;

    let maxH = 0;
    for (const item of data) {
      for (const key of info.legend_key_list) {
        maxH = Math.max(maxH, item[key]);
      }
    }
    maxH = (maxH > 0.5) ? 1 : (maxH > 0.25) ? 0.5 : 0.25;
    const yScale = d3.scaleLinear()
      .domain([0, maxH])
      //.range([height, 0]);
      .range([info.y_scale.range()[0], 0]);
    info.old_x_scale = info.x_scale;
    info.x_scale = xScale;
    info.old_y_scale = info.y_scale;
    info.y_scale = yScale;

    const animateFunc = function (_vn) {
      return (_vn.transition().duration(animate_time));
    };

    xScaleInfo.axis_func(animateFunc(info.vn.xAxis)
      .attr('transform', 'translate(0,' + height + ')'));
    animateFunc(info.vn.yAxis)
      .call(d3.axisLeft(yScale).ticks(5));
    animateFunc(info.vn.xAxisLabel)
      .attr('x', width + 6).attr('y', height + 3);
    animateFunc(info.vn.yAxisLabel)
      .attr('x', 5).attr('y', 10);
    that.updateXAxisFlags(info, undefined, xScale, undefined);
    that.updateTimeRegionBorder(info, function (_vn) {
      return (_vn.transition().delay(animate_time - 10).duration(10));
    });
  };

  that.updateTimeRegionBorder = function (info, animate_func) {
    // render border of time region
    let yAxisHeight = info.y_scale.range()[0];
    let timeRegionBorder = info.x_scale.range();
    timeRegionBorder = timeRegionBorder.slice(1, timeRegionBorder.length - 1);
    let slc = info.vn.timeRegionBorder.selectAll('line.border').data(timeRegionBorder);

    // style of line
    SVGUtil.styleD3(info.vn.timeRegionBorder, {
      stroke: 'grey',
      'stroke-width': 1,
      'stroke-dasharray': "5,5"
    });

    let slcEnter = slc.enter().append('line').classed('border', true).style('opacity', 0);
    slc.style('opacity', 0);
    slc.exit().remove();
    const _update_time_region_border = function (_slc) {
      _slc = SVGUtil.attrD3(_slc, {
        x1: (d) => (d),
        x2: (d) => (d),
        y1: 0,
        y2: yAxisHeight
      });
      return _slc;
    };
    animate_func(_update_time_region_border(slcEnter)).style('opacity', 1);
    animate_func(_update_time_region_border(slc)).style('opacity', 1);
  };

  /**
   * Latest change: Remove background bar.
   * Update flag elements on x-axis.
   * Each flag is a bar, its height encode the number of nodes in its range on x.
   */
  that.updateXAxisFlags = function (info, area, x_scale, slice_num) {
    let root = info.vn.root,
      g = root.select('g.flag-group'),
      yPadding = 5,
      xScaleRange = x_scale.range();
    if (g.empty()) {
      g = root.append('g').attr('class', 'flag-group');
      // add border
      area.x = xScaleRange[0];
      area.width = xScaleRange[xScaleRange.length - 1] - area.x;
      g.append('line')
        .attr('x1', area.x + .5)
        .attr('y1', area.y)
        .attr('x2', area.x + .5)
        .attr('y2', area.y + area.height)
        .style('stroke', 'black')
        .style('stroke-width', 1);
      /*
      g.append('line')
        .attr('x1', area.x + 1 + area.width)
        .attr('y1', area.y)
        .attr('x2', area.x + 1 + area.width)
        .attr('y2', area.y + area.height)
        .style('stroke', 'black')
        .style('stroke-width', 1);
      */
      /*
      g.append('rect')
        .attr('x', area.x)
        .attr('y', area.y + 1)
        .attr('width', area.width)
        .attr('height', area.height - 2)
        .style('fill', 'transparent')
        .style('stroke', 'gray')
        .style('stroke-width', .5);
        */
    }
    if (area === undefined) {
      area = info.box_setting.area;
      slice_num = info.box_setting.slice_num;
    }
    area.x = xScaleRange[0];
    area.width = xScaleRange[xScaleRange.length - 1] - area.x;
    info.box_setting = {
      area,
      slice_num,
      x_scale,
      normal_opacity: 0.3,
      y_padding: yPadding
    };
    let flagControl = [],
      barRange = {};
    for (let i = 0; i < slice_num; ++i) {
      let left_t = Math.round(x_scale.invert(area.x + (i / slice_num) * area.width)) + ((i > 0) ? 1 : 0),
        right_t = Math.round(x_scale.invert(area.x + (i + 1) / slice_num * area.width)),
        all_t = right_t - left_t + 1;

      flagControl.push({
        cur: i,
        all: slice_num,
        scale: x_scale,
        left_t,
        right_t,
        all_t,
      });
      barRange.max = (barRange.max === undefined) ? all_t : Math.max(barRange.max, all_t);
    }
    let realH = area.height - yPadding;
    console.log('area.height', area.height, 'ypadding', yPadding)
    let barScale = d3.scaleLinear().domain([0, barRange.max]).range([0, realH]);
    info.box_setting.bar_scale = barScale;

    let _render = function (_slc) {
      _slc.each(function (d) {
        let vn = d3.select(this),
          bar = vn.select('rect.x-axis-flag-bak');
        if (bar.empty()) {
          vn.append('rect').attr('class', 'x-axis-flag-bak').style('display', 'none');
          vn.append('rect').attr('class', 'x-axis-flag-slc');
        }
      });
    };

    let slc = g.selectAll('g.x-axis-flag').data(flagControl, (d) => (d.cur));
    _render(slc.enter().append('g').attr('class', 'x-axis-flag'));
    _render(slc);
    slc.exit().remove();

    //that.updateXAxisFocus(info, DataLoader.curData, animate_func);
  };

  /**
   * Update focused flag elements on x-axis.
   * Each flag is a bar, its height encode the number of nodes in its range on x.
   */
  that.updateXAxisFocus = function (info, data, animate_func) {
    animate_func = (animate_func === undefined) ? ((x) => (x)) : animate_func;
    let root = info.vn.root,
      g = root.select('g.flag-group');
    let area = info.box_setting.area,
      yPadding = info.box_setting.y_padding;
    //barScale = info.box_setting.bar_scale;
    let barW = area.width * (1 / info.box_setting.slice_num);
    let realH = 7;//area.height - yPadding;

    let selectedDic = data.selected_idx || (new Set());
    let showedDic = data.showed_idx || (new Set());

    let sliceList = [];
    g.selectAll('g.x-axis-flag').each((d) => {
      let selected_t_num = 0, showed_t_num = 0;
      for (let j = d.left_t; j <= d.right_t; ++j) {
        if (selectedDic.has(j)) {
          selected_t_num += 1;
        }
        if (showedDic.has(j)) {
          showed_t_num += 1;
        }
      }
      d.selected_t_num = selected_t_num;
      d.showed_t_num = showed_t_num;
      sliceList.push(d);
    });

    //// smoonth selected_t_num on average three
    //let smoonth_r = 2;
    //sliceList.forEach((slice, idx) => {
    //  if (slice.selected_t_num == 0) {
    //    slice.smoonth_value = 0;
    //    return;
    //  }
    //  let l = Math.max(0, idx - smoonth_r),
    //    r = Math.min(sliceList.length - 1, idx + smoonth_r);
    //  slice.smoonth_value = 0;
    //  for (let i = l; i <= r; ++i) {
    //    slice.smoonth_value += sliceList[i].selected_t_num;
    //  }
    //  slice.smoonth_value /= (r - l + 1);
    //});
    sliceList.forEach((slice) => {
      slice.status = 0
      if (slice.showed_t_num) {
        slice.status = 1
      }
      if (slice.selected_t_num) {
        slice.status = 3
      }
      slice.smoonth_value = slice.selected_t_num / (slice.right_t - slice.left_t + 1 + 1e-5);
    });
    const barScale = d3.scaleLinear().domain([0, 1]).range([0, realH]);

    //// normalize by the value range in expand time region of linechart
    //const regions = ActionTrail.getExpandTimeRegion();
    //let maxValue = 0,
    //  allMaxValue = 0;
    //sliceList.forEach((d) => {
    //  // check if the slice is in expand time region
    //  let flagExpand = false,
    //    lx = d.cur / d.all,
    //    rx = (d.cur + 1) / d.all;
    //  regions.forEach((r) => {
    //    if (r.l <= (d.cur - smoonth_r) / d.all && r.r >= (d.cur + smoonth_r) / d.all) {
    //      flagExpand = true;
    //    }
    //  });
    //  if (flagExpand) {
    //    maxValue = Math.max(maxValue, d.smoonth_value);
    //  }
    //  allMaxValue = Math.max(allMaxValue, d.smoonth_value);
    //});
    //if (maxValue == 0) {
    //  maxValue += 1e-2;
    //}
    //allMaxValue += 2e-2;
    //let barScale = d3.scaleLinear().domain([0, maxValue, allMaxValue]).range([0, realH, realH]);

    g.selectAll('g.x-axis-flag').each(function (d) {
      //let slcBarH = barScale(d.selected_t_num);
      let slcBarH = barScale(d.smoonth_value);
      animate_func(d3.select(this).select('rect.x-axis-flag-slc'))
        .attr('x', area.x + area.width * (d.cur / d.all) + 0.25)
        .attr('y', yPadding + area.y - realH)
        .attr('width', barW)
        .attr('height', d.status > 0 ? realH : 0)
        .style('fill', 'rgba(204, 98, 87, 0.8)')//'rgba(128, 128, 128, 0.5)')
        .style('stroke-width', 0)
        .style('fill-opacity', d => d.status * 0.3);
    });
  };

  /**
   * Update curves in linechart during the sedimentation animation.
   */
  that.updatePathInSedimentation = function (info, cur_num, animate_time) {
    const _scale = {
      x: info.x_scale,
      y: info.y_scale,
    };
    const _old_scale = {
      x: info.old_x_scale,
      y: info.old_y_scale,
    };
    const part_data = info.data.slice(0, cur_num);
    const glypher = info.glypher;
    const color = that.color;

    const glyph_shift_step = 1.0 / info.legend_key_list.length;
    for (let idx = 0; idx < info.legend_key_list.length; ++idx) {
      const _key = info.legend_key_list[idx];
      const glyphPathSetting = {
        shift: idx * glyph_shift_step,
        name: _key,
      };
      if (animate_time) {
        const pathTweenInfo = utils.pathTween.getIncreasing(info.olddata, info.data, _old_scale, _scale);
        info.legendDic[_key].path.transition().duration(animate_time)
          .attrTween('d', pathTweenInfo.axis);
        if (glypher !== undefined) {
          const tweenInfo = utils.getGlyphTween(glyphPathSetting,
            info.olddata, info.data, _old_scale, _scale);
          const tween_list = [{
            time: animate_time,
            tween_func: tweenInfo.axis, // TODO
          }];
          utils.updatePathGlyphWithTweenAnimation(
            info.legendDic[_key].glyph.path, tween_list);
        }
      } else {
        info.legendDic[_key].path.attr('d', function () {
            return d3.line()
              .x(function (d, i) {
                return _scale.x(d.t);
              })
              .y(function (d) {
                return _scale.y(d[_key]);
              })
              .curve(d3.curveMonotoneX)(part_data);
          })
          .attr('stroke', () => color(_key))
          .attr('fill', 'none');
        if (glypher !== undefined) {
          utils.updatePathGlyph(info.legendDic[_key].glyph.path, glyphPathSetting,
            part_data, glypher.get(_key), _scale);
        }
      }
    }
  };

  // support for drawing multiple line chart at different area on one svg.
  that.drawGridLinechart = function (visual_setting, data, unique_id, legend_key_list) {
    //console.log('DEBUG drawGridLinechart: number of data=', data.length);
    const container = visual_setting.container;
    let area = visual_setting.area;
    const yLabel = visual_setting.y_label;

    const color = (d) => ('black');

    const slc = container.select('#' + unique_id);
    let root = undefined;
    let info = {
      id: unique_id,
    };

    if (slc.empty()) {
      root = container.append('g').datum(info).attr('id', unique_id);
      root.attr('transform', 'translate(' + area.x + ',' + area.y + ')');
      info.area = area;
      info.vn = {};
      info.vn.root = root;
      info.vn.yLabel = root.append('text').style('font-size', that.label_font_size-1)
        .attr('x', -5).attr('y', -10).text(yLabel);
    } else {
      slc.each(function (d) {
        info = d;
      });
      root = info.vn.root;
      area = info.area;
    }

    const unitSize = {
      h: 0,
      w: 0,
    };
    const num_per_line = 2;
    const padding = 2;
    if (that.feature_view_width === undefined) {
      that.feature_view_width = d3.select('#feature-grid-view').node().getBoundingClientRect().width - 25;
      d3.select('#feature-grid-svg').attr('width', that.feature_view_width);
    }
    unitSize.w = (that.feature_view_width - padding) / num_per_line - padding - 1;
    unitSize.h = 0.618 * unitSize.w;
    if (that.feature_view_height === undefined) {
      that.feature_view_height = Math.ceil(legend_key_list.length / num_per_line) * (unitSize.h + padding) + padding;
      d3.select('#feature-grid-svg').attr('height', that.feature_view_height);
    }

    const _drawLinechartOverview = function (_vn, _key, _area, _is_selected) {
      // background
      let box = SVGUtil.ensureSpecificSVGItem(_vn, 'rect', 'gridLC-ow-bk');
      box = utils.attrD3(box, _area);
      box = utils.styleD3(box, {
        'fill': 'transparent',
        'stroke-width': (_is_selected ? 1.5 : .75),
        'stroke': (_is_selected ? 'rgb(204,98,87)' : 'lightgrey'),
      });

      // path
      const lcImage = SVGUtil.ensureSpecificSVGItem(_vn, 'image', 'gridLC-ow');
      SVGUtil.attrD3(lcImage, {
        x: _area.x + 1,
        y: _area.y + 1,
        width: _area.width - 2,
        height: _area.height - 2,
      });
      const lcImageCallback = function (_static_image_path) {
        lcImage.attr('xlink:href', ActionTrail.static_path + _static_image_path);
      };
      const lineData = [data.map((d) => (d[_key]))];
      //if (_key == 'ED') {
      //  lineData.push(data.map((d) => (d['ED2'])));
      //}
      utils.saveLineChartAsSVG({
        w: 500,
        h: 500 * (_area.height / _area.width),
        color: color(_key),
      }, lineData, lcImageCallback);

      // attr_name
      let text = SVGUtil.ensureSpecificSVGItem(_vn, 'text', 'gridLC-ow-text');
      text = utils.attrD3(text, {
        'transform': `translate(${_area.width / 2}, 15)`,
        'text-anchor': 'middle',
      });
      text.text(that.convert_legend_name_in_grid(_key));

      // over & out
      _vn.on('mouseover', function (d) {
        utils.styleD3(box, {
          'stroke-width': 2,
        });
      }).on('mouseout', function (d) {
        utils.styleD3(box, {
          'stroke-width': 1,
        });
      });
    };

    const _drawUnit = function (_slc) {
      _slc.each(function (d) {
        const row = Math.floor(order[d] / num_per_line + 1e-8);
        if (unitSize.h * row > that.feature_view_height) {
          return;
        }
        const col = order[d] - row * num_per_line;
        const vn = d3.select(this);
        const ox = col * unitSize.w;
        const oy = row * unitSize.h;
        const padding = 1;
        vn.attr('transform', 'translate(' + ox + ',' + oy + ')');
        _drawLinechartOverview(vn, d, {
          x: padding,
          y: padding,
          width: unitSize.w - padding * 2,
          height: unitSize.h - padding * 2,
        }, (isSelected[d] !== undefined));
      }).on('click', function (d) {
        if (isSelected[d] === undefined) {
          // window.ActionTrail.notify('select-feature-detail');
          //if (d == 'ED') {
          //  DataLoader.selected_feature_keys.push('ED2');
          //}
          DataLoader.selected_feature_keys.push(d);
          DataLoader.update_line_plot();
        } else {
          // window.ActionTrail.notify('remove-feature-detail');
          const new_arr = [];
          DataLoader.selected_feature_keys.forEach((dd, i) => {
            if (i == isSelected[d]) return;
            //if (d == 'ED' && i == isSelected['ED2']) {
            //  return;
            //}
            new_arr.push(dd);
          });
          DataLoader.selected_feature_keys = new_arr;
          DataLoader.update_line_plot();
        }
      });
    };

    const order = {};
    legend_key_list.forEach((d, i) => {
      order[d] = i;
    });

    const isSelected = {};
    DataLoader.selected_feature_keys.forEach((key, i) => {
      isSelected[key] = i;
    });
    const unitSlc = root.selectAll('g.grid-linechart-unit').data(legend_key_list, (d) => (d));
    _drawUnit(unitSlc.enter().append('g').attr('class', 'grid-linechart-unit'));
    _drawUnit(unitSlc);
    unitSlc.exit().remove();
  };

  that.getCustomTicks = function (scale, width) {
    let tickValues = [],
      tickTag = {},
      win_size = DataLoader.win_size;
    for (let tim = scale.invert(width); tim >= 0; tim -= win_size) {
      tickValues.push(tim);
    }
    let _tag = DataLoader.data[0].timestamp.split(' ')[0];
    // the width calculated using canvas seems a little different from svg text.
    // my previous solution is using http://github.com/Pomax/Font.js
    // let font = new Font();
    // font.fontFamily = xxx;
    // font.src = xxx;
    // font.loadFont(); // it is async!
    // then use font.measureText(word, font_size).width;
    let tickMinBlock = getTextWidth(_tag, "11px Arial") + 4,
      preTagIndex = undefined,
      minTick = undefined;

    for (let i = 0; i < tickValues.length; ++i) {
      let tim = tickValues[i];
      if (preTagIndex === undefined || (scale(preTagIndex) - scale(tim) > tickMinBlock &&
          scale(tim) - scale(0) > tickMinBlock)) {
        let tag = DataLoader.data[tim].timestamp.split(' ')[0];
        tickTag[tim] = tag;
        preTagIndex = tim;
        minTick = scale(tim);
      } else {
        tickTag[tim] = '';
      }
    }
    if (tickValues[tickValues.length - 1] > 0 && minTick - scale(0) > tickMinBlock) {
      tickValues.push(0);
      tickTag[0] = DataLoader.data[0].timestamp.split(' ')[0];
    }

    tickValues.reverse();
    let lastyear = '';
    for (let i = 0; i < tickValues.length; ++i) {
      let tim = tickValues[i];
      let tag = tickTag[tim];
      if (tag.split('-')[0] == lastyear) {
        tickTag[tim] = tag.slice(tag.indexOf('-') + 1)
      } else if (tag.length > 0 && tag.indexOf('-') != -1) {
        lastyear = tag.split('-')[0]
      }
    }

    return {
      tickValues,
      tickTag
    };
  };

  that.getSegmentXScale = function (minx, maxx, datalen = -1) {
    const _lastone = (_arr) => (_arr[_arr.length - 1]),
      width = maxx - minx,
      time_regions = window.ActionTrail.timeline,
      data_nodes = window.ActionTrail.dataNodes;
    let count_t = [0],
      count_x = [minx];
    for (let r of time_regions) {
      let curDataLen = _lastone(data_nodes[_lastone(r.idxs)].newNodes)
      if (datalen != -1) curDataLen = Math.min(curDataLen, datalen)
      count_t.push(curDataLen);
      count_x.push(minx + r.x * width);
    }
    const scale = d3.scaleLinear()
      .domain(count_t)
      .range(count_x);
    const {
      tickValues,
      tickTag
    } = that.getCustomTicks(scale, width);
    const axis_func = function (g) {
      g.call(d3.axisBottom(scale)
          .tickValues(tickValues)
          .tickFormat((d) => (tickTag[d])))
        //.tickFormat((d) => (DataLoader.data[d].timestamp.split(' ')[0])))
        .selectAll('text')
        //.style("text-anchor", "end")
        .style("text-anchor", "middle")
        .attr("dx", //d => tickTag[d].length > 5 ? "2em": "0em")
          function (d) {
            return 0;
            //return (d3.select(this).text().length > 5 ? "2em" : "0em");
          })
        .attr("dy", ".7em")
      //.attr("transform", "rotate(-30)");
    };
    return {
      scale,
      axis_func
    };
  };

  const utils = {};
  utils.svg_prefix = 'linechart-related-';
  utils.svg_no = 0;
  utils.attrD3 = function (vnode, info) {
    for (const x in info) {
      vnode = vnode.attr(x, info[x]);
    }
    return vnode;
  };

  utils.styleD3 = function (vnode, info) {
    for (const x in info) {
      vnode = vnode.style(x, info[x]);
    }
    return vnode;
  };

  utils.saveLineChartAsSVG = function (settings, data, callback) {
    let paddingY = 20;
    utils.svg_no += 1;
    const svgInfo = {};
    svgInfo.no = utils.svg_prefix + utils.svg_no;
    svgInfo.area = {
      x: '0',
      y: '0',
      w: '' + settings.w,
      h: '' + settings.h,
    };

    // get svg content
    const maxXAxis = data[0].length - 1;
    const xScale = d3.scaleLinear()
      .domain([0, maxXAxis])
      .range([0, settings.w]);
    const yScale = d3.scaleLinear()
      .domain([-0.3, 1])
      .range([settings.h - paddingY, 0]);
    const tempSVG = d3.select('body').append('svg').style('display', 'none');
    const data2line = d3.line()
      .x(function (d, i) {
        return xScale(i);
      })
      .y(function (d) {
        return yScale(d);
      })
      .curve(d3.curveMonotoneX);
    tempSVG.selectAll('path').data(data).enter().append('path')
      //tempSVG.append('path')
      .attr('d', (d) => data2line(d))
      .style('stroke-width', 2)
      .style('stroke', settings.color)
      .style('fill', 'none');
    svgInfo.svg = tempSVG.html();
    tempSVG.remove();

    const req = new request_node(ActionTrail.save_svg_url, (data) => {
      callback(data.static_svg_path);
    }, 'json', 'POST');
    req.set_header({
      'Content-Type': 'application/json;charset=UTF-8',
    });
    req.set_data(svgInfo);
    req.notify();
  };

  utils.updateLineGlyph = function (_vn, _glyph, _xy) {
    return SVGUtil.CustomGlyph[_glyph](_vn,
      _xy,
      // {
      //    x: width - 45,
      //    y: legend_y - 15
      // },
      3);
  };

  utils.__sampleLinechart = function (_arr, num, shift) {
    shift = (shift === undefined) ? 0 : shift;
    const new_data = [];
    // step = Math.max(1, Math.floor(_arr.length / num));
    const step = _arr.length / num;
    // for (let i = _arr.length - 1 - Math.floor(shift * step); i >= step / 2; i -= step) {
    for (let step_i = 0; step_i < num; ++step_i) {
      const i = Math.ceil(_arr.length - Math.floor(shift * step) - step * step_i) - 1;
      new_data.push({
        id: i,
        data: _arr[i],
        indexInSeq: new_data.length,
      });
    }
    new_data.forEach((d) => {
      d.num = new_data.length;
    });
    return new_data.reverse();
  };

  utils.sampleLinechart = function (_arr, num, shift, _scale) {
    shift = (shift === undefined) ? 0 : shift;
    const new_data = [];
    // step = Math.max(1, Math.floor(_arr.length / num));
    const step = 1.0 / num,
      xr = [_scale.x(_arr[0].t), _scale.x(_arr[_arr.length-1].t)];
    for (let step_i = 0; step_i < num; ++step_i) {
      const i = Math.round(_scale.x.invert(xr[1] - (xr[1]-xr[0]) * (shift+step) * step_i));
      if (i<0) {
        break;
      }
      new_data.push({
        id: i,
        data: _arr[i],
        indexRate: 1-(shift+step)*step_i,
        indexInSeq: new_data.length,
      });
    }
    new_data.forEach((d) => {
      d.num = new_data.length;
    });
    return new_data.reverse();
  };

  utils.updatePathGlyph = function (_vn, _key_info, _data, _glyph, _scale) {
    const sampledData = utils.sampleLinechart(_data, 10, _key_info.shift, _scale);
    const _key = _key_info.name;
    const gSlc = _vn.selectAll('g.lc-glyph')//.data(sampledData, (d) => (d.indexInSeq));
      .data(sampledData, (d) => (d.indexRate))
    const _draw = function (x) {
      x.each(function (d) {
        d3.select(this).attr('transform', 'translate(' + (
          _scale.x(d.id)) + ',' + (_scale.y(d.data[_key])) + ')');
        SVGUtil.CustomGlyph[_glyph](d3.select(this), {
            x: 0, // _scale.x(d.id),
            y: 0, // _scale.y(d.data[_key])
          },
          3);
      });
    };
    _draw(gSlc.enter().append('g').attr('class', 'lc-glyph'));
    _draw(gSlc);
    gSlc.exit().remove();
    return _vn;
  };

  utils.updatePathGlyphWithTweenAnimation = function (_vn, _tween_list) {
    _vn = _vn.selectAll('g.lc-glyph');
    for (const _tween of _tween_list) {
      _vn = _vn.transition().duration(_tween.time)
        .attrTween('transform', _tween.tween_func);
    }
    return _vn;
  };

  utils.updatePath = function (_vn, _data, _scale) {
    return _vn.attr('d', function (_key) {
        return d3.line()
          .x(function (d, i) {
            return _scale.x(d.t);
          })
          .y(function (d) {
            return _scale.y(d[_key]);
          })
          .curve(d3.curveMonotoneX)(_data);
      })
      .attr('stroke', (_key) => that.color(_key))
      .attr('fill', 'none');
  };

  utils.interpolateTwoSegmentScale = function (oldScale, newScale) {
    let oldRange = oldScale.range(),
      oldDomain = oldScale.domain(),
      newDomain = newScale.domain();
    let newDomainMax = newDomain[newDomain.length - 1];
    const oldRangeChanges = [];
    for (let i = 0; i < oldRange.length; ++i) {
      if (oldDomain[i] > newDomainMax) {
        break;
      }
      oldRangeChanges.push(d3.scaleLinear()
        .domain([0, 1]).range([oldRange[i], newScale(oldDomain[i])]));
    }
    const interpolateScale = function (t) {
      let tempRange = [];
      for (let change of oldRangeChanges) {
        tempRange.push(change(t));
      }
      return d3.scaleLinear().domain(oldDomain).range(tempRange);
    };

    return interpolateScale;
  };

  utils.pathTween = {};
  utils.pathTween.animateYScaleTween = function (oldScale, newScale) {
    const animateYScale = d3.scaleLinear().domain([0, 1])
      .range([oldScale.y.domain()[1], newScale.y.domain()[1]]);
    const interpolateY = function (t) {
      let tempScale = d3.scaleLinear().domain([0, animateYScale(t)])
        .range(newScale.y.range());
      return tempScale;
    };
    return interpolateY;
  };

  utils.pathTween.getIncreasing = function (oldData, newData, oldScale, newScale) {
    function _curve(_key) {
      const _line = d3.line()
        .x(function (d, i) {
          return newScale.x(d.t);
        })
        .y(function (d) {
          return newScale.y(d[_key]);
        });
      let interpolate = d3.scaleQuantile()
        .domain([0, 1])
        .range(d3.range(oldData.length, newData.length));
      if (newData.length == oldData.length) {
        interpolate = function () {
          return newData.length;
        };
      }
      return function (t) {
        return _line(newData.slice(0, interpolate(t)));
      };
    };

    const interpolateY = utils.pathTween.animateYScaleTween(oldScale, newScale);

    function _axis(_key) {
      const interpolateScale = utils.interpolateTwoSegmentScale(oldScale.x, newScale.x);

      return function (t) {
        const _temp_scale_x = interpolateScale(t);
        const _line = d3.line()
          .x(function (d, i) {
            return _temp_scale_x(d.t);
          })
          .y(function (d) {
            //return oldScale.y(d[_key]);
            return interpolateY(t)(d[_key]);
          });
        return _line(oldData);
      };
    };

    return {
      axis: _axis,
      curve: _curve,
    };
  };

  utils.pathTween.getDecreasing = function (oldData, newData, oldScale, newScale) {
    function _curve(_key) {
      const _line = d3.line()
        .x(function (d, i) {
          return oldScale.x(d.t);
        })
        .y(function (d) {
          return oldScale.y(d[_key]);
        });
      let interpolate = d3.scaleQuantile()
        .domain([0, 1])
        .range(d3.range(newData.length, oldData.length).reverse());
      if (newData.length == oldData.length) {
        interpolate = function () {
          return newData.length;
        };
      }
      return function (t) {
        return _line(oldData.slice(0, interpolate(t)));
      };
    };

    const interpolateY = utils.pathTween.animateYScaleTween(oldScale, newScale);

    function _axis(_key) {
      const interpolateScale = utils.interpolateTwoSegmentScale(oldScale.x, newScale.x);

      return function (t) {
        const _temp_scale_x = interpolateScale(t);
        const _line = d3.line()
          .x(function (d, i) {
            return _temp_scale_x(d.t);
          })
          .y(function (d) {
            //return newScale.y(d[_key]);
            return interpolateY(t)(d[_key]);
          });
        return _line(newData);
      };
    };

    return {
      axis: _axis,
      curve: _curve,
    };
  };

  utils.getGlyphTween = function (_key_info, oldData, newData, oldScale, newScale) {
    const _key = _key_info.name;
    const interpolateScale = utils.interpolateTwoSegmentScale(oldScale.x, newScale.x);
    const interpolateY = utils.pathTween.animateYScaleTween(oldScale, newScale);

    function _axis(d) {
      return function (t) {
        const _temp_scale_x = interpolateScale(t);
        return ('translate(' +
          (_temp_scale_x(d.id)) + ',' +
          (interpolateY(t)(d.data[_key])) + ')');
      };
    }

    let minCtx = {
        data: oldData,
        scale: oldScale
      },
      maxCtx = {
        data: newData,
        scale: newScale
      };
    if (oldData.length >= newData.length) {
      let temp = minCtx;
      minCtx = maxCtx;
      maxCtx = temp;
    }
    let changeMaxT = d3.range(minCtx.data.length, maxCtx.data.length);
    if (oldData.length >= newData.length) {
      changeMaxT = changeMaxT.reverse();
    }

    function _curve(d) {
      let interpolate = d3.scaleQuantile()
        .domain([0, 1])
        .range(changeMaxT);
      if (minCtx.data.length == maxCtx.data.length) {
        interpolate = function () {
          return maxCtx.data.length - 1;
        };
      }
      return function (t) {
        // d.id = Math.round(interpolate(t) * (d.indexInSeq / d.num));
        //d.id = Math.floor(interpolate(t) * (1 - d.indexInSeq / d.num - 1.0 / d.num * _key_info.shift));
        const _sx = maxCtx.scale.x;
        //if (d.indexRate < 1e-2)
        //  console.log('---------- func', _sx.invert(_sx(interpolate(t))*d.indexRate));
        d.id = Math.floor(_sx.invert(_sx(interpolate(t))*d.indexRate));
        d.data = maxCtx.data[d.id];
        return ('translate(' +
          (maxCtx.scale.x(d.id)) + ',' +
          (maxCtx.scale.y(d.data[_key])) + ')');
        //(interpolateY(t)(d.data[_key])) + ')');
      };
    }
    return {
      axis: _axis,
      curve: _curve,
    };
  };

})(window);