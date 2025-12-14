from PyQt5 import QtWidgets, QtCore, QtGui


class RowReorderTable(QtWidgets.QTableWidget):
    """
    QTableWidget 子类：支持整行拖拽重排。
    在 dropEvent 中把整表复制为列表，重排后重建表格内容，
    并在完成后发射 rowsReordered 信号。
    """
    rowsReordered = QtCore.pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 允许内部移动
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setDropIndicatorShown(True)

    def dropEvent(self, event):
        # 1. 收集当前表的所有行第一列的文本
        total_rows = self.rowCount()
        rows = []
        for r in range(total_rows):
            it = self.item(r, 0)
            if it is not None and it.text().strip() != "":
                rows.append(it.text())

        # 2. 找到被选中的行索引（按当前 selection）
        selected_rows = sorted({idx.row() for idx in self.selectedIndexes()})
        if not selected_rows:
            # 没有选中则退回默认处理（避免无意行为）
            super().dropEvent(event)
            return

        # 强制只处理单行：优先使用 currentRow()（用户当前行），否则使用选中的第一个
        current = self.currentRow()
        if current is None or current < 0:
            sel = selected_rows[0]
        else:
            # 如果 currentRow 在选中集中就用它，否则也退回到第一个选中
            sel = current if current in selected_rows else selected_rows[0]
        selected_rows = [sel]

        # 3. 计算 drop 目标位置（原表坐标）
        drop_index = self.indexAt(event.pos()).row()
        if drop_index == -1:
            # 如果是拖到表格空白处，设为末尾（插入在最后）
            drop_index = total_rows

        # 4. 从 rows 中分离出被移动的条目和剩余条目
        moved = [rows[i] for i in selected_rows]
        remaining = [rows[i] for i in range(len(rows)) if i not in selected_rows]

        # 5. 计算在 remaining 中的插入位置（考虑原 drop_index 前有多少被选行）
        num_before = sum(1 for i in selected_rows if i < drop_index)
        insert_at = max(0, drop_index - num_before)
        if insert_at > len(remaining):
            insert_at = len(remaining)

        # 6. 生成新的行序列
        new_rows = remaining[:insert_at] + moved + remaining[insert_at:]

        # 9. 发射信号通知外部（StatusPanel）更新内部数据结构顺序
        self.rowsReordered.emit(new_rows)
        self.setFocus(QtCore.Qt.MouseFocusReason)
        event.accept()

    def startDrag(self, supportedActions):
        """
        覆写 startDrag：使用 QDrag 手动执行拖放，并将默认动作设为 CopyAction。
        这样拖到外部目标（如浏览器）时只做复制，不会导致源表格行被移除/清空。
        对内部拖拽（表格内部重排）仍然支持 Move（dropEvent 的逻辑会处理行顺序）。
        """
        # 取选中索引并生成 mimeData（与 Qt 内部一致）
        mime = self.model().mimeData(self.selectedIndexes())
        drag = QtGui.QDrag(self)
        drag.setMimeData(mime)
        # 执行拖放：
        # - 支持 Copy | Move 动作，
        # - 默认（第二个参数）设置为 CopyAction，保证外部 drop 使用 copy semantics（不会移除源）
        drag.exec(QtCore.Qt.CopyAction | QtCore.Qt.MoveAction, QtCore.Qt.CopyAction)
        return

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        """
        处理拖拽移动事件，实现靠近边缘时的自动滚动。
        """
        # 允许父类（QTableWidget）处理拖放指标（插入线）的更新
        super().dragMoveEvent(event)
        scroll_bar = self.verticalScrollBar()

        # 定义触发自动滚动的边缘区域大小（像素）
        SCROLL_MARGIN = 30
        # 定义每次滚动的步长（设置为单步的 2 倍，以加快滚动速度）
        SCROLL_STEP = scroll_bar.singleStep() * 2

        # 1. 检查是否靠近顶部边缘
        if event.pos().y() < self.viewport().rect().top() + SCROLL_MARGIN:
            current_value = scroll_bar.value()
            # 向上滚动
            if current_value > scroll_bar.minimum():
                scroll_bar.setValue(current_value - SCROLL_STEP)
                event.accept()
                return

        # 2. 检查是否靠近底部边缘
        if event.pos().y() > self.viewport().rect().bottom() - SCROLL_MARGIN:
            current_value = scroll_bar.value()
            # 向下滚动
            if current_value < scroll_bar.maximum():
                scroll_bar.setValue(current_value + SCROLL_STEP)
                event.accept()
                return

        # 如果没有触发自动滚动，继续接受事件以保持拖拽指示器可见
        event.accept()
