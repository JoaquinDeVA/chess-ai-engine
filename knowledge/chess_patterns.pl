% chess_patterns.pl
% Prolog knowledge base for tactical pattern detection in chess

% Dynamic predicates that will be asserted from Python
:- dynamic piece/4.        % piece(Type, Color, Row, Col)
:- dynamic move/4.         % move(FromRow, FromCol, ToRow, ToCol)

% Define opposite colors
opposite_color(white, black).
opposite_color(black, white).

% Piece values (typical values, king has very high value)
piece_value(p, 1).      % Pawn
piece_value(n, 3).      % Knight
piece_value(b, 3).      % Bishop
piece_value(r, 5).      % Rook
piece_value(q, 9).      % Queen
piece_value(k, 100).    % King

% MVV-LVA piece values (for capture scoring, king has practical value)
mvv_lva_piece_value(p, 1).  % Pawn
mvv_lva_piece_value(n, 3).  % Knight
mvv_lva_piece_value(b, 3).  % Bishop
mvv_lva_piece_value(r, 5).  % Rook
mvv_lva_piece_value(q, 9).  % Queen
mvv_lva_piece_value(k, 10). % King

% Square relationships
same_row(R, R).
same_col(C, C).

same_diagonal(R1, C1, R2, C2) :-
    Diff1 is R2 - R1,
    Diff2 is C2 - C1,
    abs(Diff1) =:= abs(Diff2).

% Line relationships (for pins and skewers)
on_same_line(R1, C1, R2, C2, R3, C3) :-
    same_row(R1, R2),
    same_row(R2, R3),
    ((C1 < C2, C2 < C3); (C3 < C2, C2 < C1)).

on_same_line(R1, C1, R2, C2, R3, C3) :-
    same_col(C1, C2),
    same_col(C2, C3),
    ((R1 < R2, R2 < R3); (R3 < R2, R2 < R1)).

on_same_line(R1, C1, R2, C2, R3, C3) :-
    same_diagonal(R1, C1, R2, C2),
    same_diagonal(R2, C2, R3, C3),
    ((R1 < R2, R2 < R3, C1 < C2, C2 < C3);
     (R3 < R2, R2 < R1, C3 < C2, C2 < C1);
     (R1 < R2, R2 < R3, C1 > C2, C2 > C3);
     (R3 < R2, R2 < R1, C3 > C2, C2 > C1)).

% Generate intermediate squares on a diagonal between two points
% diagonal_between(R1, C1, R2, C2, Ri, Ci)
% Works like between/3 but for diagonal coordinates
diagonal_between(R1, C1, R2, C2, Ri, Ci) :-
    same_diagonal(R1, C1, R2, C2),
    DiffR is R2 - R1,
    DiffC is C2 - C1,
    abs(DiffR) =:= abs(DiffC),
    Distance is abs(DiffR),
    StepR is sign(DiffR),
    StepC is sign(DiffC),
    % Generate step numbers from 1 to Distance-1 (exclude endpoints)
    between(1, Distance, Step),
    Step < Distance,
    % Calculate intermediate square
    Ri is R1 + Step * StepR,
    Ci is C1 + Step * StepC.

% Piece movement capabilities
can_attack_along_line(r, row).
can_attack_along_line(r, col).
can_attack_along_line(q, row).
can_attack_along_line(q, col).
can_attack_along_line(q, diagonal).
can_attack_along_line(b, diagonal).

% Check if path is clear between two squares
path_clear(R1, C1, R2, C2) :-
    R1 =:= R2, !,  % Same row
    C3 is min(C1, C2) + 1,
    C4 is max(C1, C2) - 1,
    \+ (between(C3, C4, C), piece(_, _, R1, C)).

path_clear(R1, C1, R2, C2) :-
    C1 =:= C2, !,  % Same column
    R3 is min(R1, R2) + 1,
    R4 is max(R1, R2) - 1,
    \+ (between(R3, R4, R), piece(_, _, R, C1)).

path_clear(R1, C1, R2, C2) :-
    same_diagonal(R1, C1, R2, C2),
    \+ (diagonal_between(R1, C1, R2, C2, Ri, Ci),
        piece(_, _, Ri, Ci)).


% Basic attack patterns for different pieces
can_attack(p, white, R, C, R2, C2) :-
    R2 is R + 1,
    (C2 is C + 1; C2 is C - 1),
    piece(_, black, R2, C2).

can_attack(p, black, R, C, R2, C2) :-
    R2 is R - 1,
    (C2 is C + 1; C2 is C - 1),
    piece(_, white, R2, C2).

can_attack(n, _, R, C, R2, C2) :-
    knight_move(R, C, R2, C2).

can_attack(b, _, R1, C1, R2, C2) :-
    same_diagonal(R1, C1, R2, C2),
    path_clear(R1, C1, R2, C2).

can_attack(r, _, R, C, R2, C2) :-
    (same_row(R, R2); same_col(C, C2)),
    path_clear(R, C, R2, C2).

can_attack(q, _, R1, C1, R2, C2) :-
    (same_row(R1, R2); same_col(C1, C2); same_diagonal(R1, C1, R2, C2)),
    path_clear(R1, C1, R2, C2).

can_attack(k, _, R, C, R2, C2) :-
    abs(R2 - R) =< 1,
    abs(C2 - C) =< 1,
    (R, C) \= (R2, C2).

% Knight movement pattern
knight_move(R1, C1, R2, C2) :-
    DR is abs(R2 - R1),
    DC is abs(C2 - C1),
    ((DR =:= 2, DC =:= 1); (DR =:= 1, DC =:= 2)).

% Helper: Check if path is clear between two squares, ignoring a specific square
path_clear_ignoring(R1, C1, R2, C2, IgnoreR, IgnoreC) :-
    R1 =:= R2, !,  % Same row
    C3 is min(C1, C2) + 1,
    C4 is max(C1, C2) - 1,
    \+ (between(C3, C4, C),
        piece(_, _, R1, C),
        (R1, C) \= (IgnoreR, IgnoreC)).

path_clear_ignoring(R1, C1, R2, C2, IgnoreR, IgnoreC) :-
    C1 =:= C2, !,  % Same column
    R3 is min(R1, R2) + 1,
    R4 is max(R1, R2) - 1,
    \+ (between(R3, R4, R),
        piece(_, _, R, C1),
        (R, C1) \= (IgnoreR, IgnoreC)).

path_clear_ignoring(R1, C1, R2, C2, IgnoreR, IgnoreC) :-
    same_diagonal(R1, C1, R2, C2),
    \+ (diagonal_between(R1, C1, R2, C2, Ri, Ci),
        piece(_, _, Ri, Ci),
        (Ri, Ci) \= (IgnoreR, IgnoreC)).

% Helper: Check if a piece at a given square can attack a target, ignoring a specific square
can_attack_ignoring(PieceType, Color, FromR, FromC, ToR, ToC, _IgnoreR, _IgnoreC) :-
    PieceType = p, Color = white, !,
    ToR is FromR + 1,
    (ToC is FromC + 1; ToC is FromC - 1),
    piece(_, black, ToR, ToC).

can_attack_ignoring(PieceType, Color, FromR, FromC, ToR, ToC, _IgnoreR, _IgnoreC) :-
    PieceType = p, Color = black, !,
    ToR is FromR - 1,
    (ToC is FromC + 1; ToC is FromC - 1),
    piece(_, white, ToR, ToC).

can_attack_ignoring(PieceType, _, FromR, FromC, ToR, ToC, _IgnoreR, _IgnoreC) :-
    PieceType = n, !,
    knight_move(FromR, FromC, ToR, ToC).

can_attack_ignoring(PieceType, _, FromR, FromC, ToR, ToC, IgnoreR, IgnoreC) :-
    PieceType = b, !,
    same_diagonal(FromR, FromC, ToR, ToC),
    path_clear_ignoring(FromR, FromC, ToR, ToC, IgnoreR, IgnoreC).

can_attack_ignoring(PieceType, _, FromR, FromC, ToR, ToC, IgnoreR, IgnoreC) :-
    PieceType = r, !,
    (same_row(FromR, ToR); same_col(FromC, ToC)),
    path_clear_ignoring(FromR, FromC, ToR, ToC, IgnoreR, IgnoreC).

can_attack_ignoring(PieceType, _, FromR, FromC, ToR, ToC, IgnoreR, IgnoreC) :-
    PieceType = q, !,
    (same_row(FromR, ToR); same_col(FromC, ToC); same_diagonal(FromR, FromC, ToR, ToC)),
    path_clear_ignoring(FromR, FromC, ToR, ToC, IgnoreR, IgnoreC).

can_attack_ignoring(PieceType, _, FromR, FromC, ToR, ToC, _IgnoreR, _IgnoreC) :-
    PieceType = k,
    abs(ToR - FromR) =< 1,
    abs(ToC - FromC) =< 1,
    (FromR, FromC) \= (ToR, ToC).

%==============================================================================
% MOVE-BASED PATTERN DETECTION
% These predicates check if a move creates a tactical pattern without
% actually updating the board state. They assume the piece exists at the
% destination square while ignoring interference from the source square.
%==============================================================================

% Check if a move creates an absolute pin and return its score
% An absolute pin occurs when a piece cannot move without exposing the king to check
move_creates_absolute_pin(FromR, FromC, ToR, ToC, PinScore) :-

    % 0. Must be a valid move
    move(FromR, FromC, ToR, ToC),

	% 1. Obtenemos la pieza atacante
    piece(PinnerType, PinnerColor, FromR, FromC),

    % 2. Verificamos que ataca en linea
    can_attack_along_line(PinnerType, _),

    % 3. Colores Contrarios
    opposite_color(PinnerColor, EnemyColor),

    % 4. Las piezas enemigas
    piece(PinnedType, EnemyColor, PinnedR, PinnedC),
    piece(k, EnemyColor, KingR, KingC),

    % 5. Que son piezas diferentes
    (PinnedR \= KingR ; PinnedC \= KingC),

    % 6. Que están en la misma linea
    on_same_line(ToR,ToC, PinnedR,PinnedC, KingR, KingC),

    % 7. Caminos despejados
    can_attack_ignoring(PinnerType, PinnerColor, ToR, ToC, PinnedR, PinnedC, FromR, FromC),
    can_attack_ignoring(PinnerType, PinnerColor, PinnedR, PinnedC, KingR, KingC, FromR, FromC),

    % 8. Valores de las piezas
    piece_value(PinnerType, PinnerValue),
    piece_value(PinnedType, PinnedValue),
    piece_value(k, KingValue),

    % 9. Calculo del valor
    PinScore is (PinnedValue + KingValue - PinnerValue) // 2.



% Check if a move creates a relative pin and return its score
% A relative pin occurs when a piece cannot move without exposing a more valuable piece
% Simplifying assumptions: TargetValue > PinnedValue AND TargetValue > PinnerValue
move_creates_relative_pin(FromR, FromC, ToR, ToC, PinScore) :-

    % 0. Must be a valid move
    move(FromR, FromC, ToR, ToC),

    % 1. Obtenemos la pieza atacante
    piece(PinnerType, PinnerColor, FromR, FromC),

    % 2. Verificamos que ataca en linea
    can_attack_along_line(PinnerType, _),


    % 3. Colores Contrarios
    opposite_color(PinnerColor, EnemyColor),

    % 4. Las piezas enemigas
    piece(PinnedType, EnemyColor, PinnedR, PinnedC),
    piece(TargetType, EnemyColor, TargetR, TargetC),

    % 5. Que son piezas diferentes
    (PinnedR \= TargetR ; PinnedC \= TargetC),

    % 6. Que están en la misma linea
    on_same_line(ToR,ToC, PinnedR,PinnedC, TargetR, TargetC),

    % 7. Caminos despejados
    can_attack_ignoring(PinnerType, PinnerColor, ToR, ToC, PinnedR, PinnedC, FromR, FromC),
    can_attack_ignoring(PinnerType, PinnerColor, PinnedR, PinnedC, TargetR, TargetC, FromR, FromC),

    % 8. Valores de las piezas
    piece_value(PinnerType, PinnerValue),
    piece_value(PinnedType, PinnedValue),
    piece_value(TargetType, TargetValue),

    % 9. Paso adicional, diferencias de valor
    TargetValue > PinnedValue,
    TargetValue > PinnerValue,


    % 10. Calculo del valor
    PinScore is (PinnedValue + TargetValue - PinnerValue) // 2.





% Check if a move creates a fork and return its score
% A fork occurs when the moved piece attacks two or more valuable opponent pieces
move_creates_fork(FromR, FromC, ToR, ToC, ForkScore) :-

    % 0. Must be a valid move
    move(FromR, FromC, ToR, ToC),

	% 1. Obtenemos la pieza atacante
    piece(AttackerType, AttackerColor, FromR, FromC),

    % 2. Colores Contrarios
    opposite_color(AttackerColor, EnemyColor),

    % 3. Las piezas enemigas
    piece(Target1Type, EnemyColor, Target1R, Target1C),
    piece(Target2Type, EnemyColor, Target2R, Target2C),

    % 4. Vemos que son piezas diferentes
    (Target1R \= Target2R ; Target1C \= Target2C),

    % 5. Verificar piezas valiosas
    member(Target1Type, [k,q,r]),
    member(Target2Type, [k,q,r]),

    % 6. Verificamos que puede atacar a ambos
    can_attack_ignoring(AttackerType, AttackerColor, ToR, ToC, Target1R, Target1C, FromR, FromC),
    can_attack_ignoring(AttackerType, AttackerColor, ToR, ToC, Target2R, Target2C, FromR, FromC),

    % 7. Obtenemos los valores de las piezas
    piece_value(Target1Type, Target1Value),
    piece_value(Target2Type, Target2Value),
    piece_value(AttackerType, AttackerValue),

    % 8. Calculamos el valor del Fork
    ForkScore is Target1Value + Target2Value - AttackerValue.




% Check if a move creates a skewer and return its score
% A skewer forces a valuable piece to move, exposing a less valuable piece behind it
move_creates_skewer(FromR, FromC, ToR, ToC, SkewerScore) :-

    % 0. Must be a valid move
    move(FromR, FromC, ToR, ToC),

    % 1. BINDING: Obtener el tipo de pieza atacante
    piece(AttackerType, Color, FromR, FromC),

    % 2. La pieza debe poder atacar en línea (reina, alfil o torre)
    can_attack_along_line(AttackerType, _),

    % 3. BINDING: Buscar dos piezas enemigas en línea
    piece(FrontType, TargetColor, R2, C2),
    piece(BehindType, TargetColor, R3, C3),

    % 4. Verificar que no sean la misma pieza
    (R2 \= R3 ; C2 \= C3),

    % 5. Verificar que son de colores opuestos
    opposite_color(Color, TargetColor),

    % 6. Verificar que la pieza es valiosa. Fíjate en la sintaxis para añadir un OR en prolog: ; significa OR
    (FrontType = k; FrontType = q; FrontType = r),

    % 7. Verificar alineación: destino -> pieza frontal -> pieza detrás
    on_same_line(ToR, ToC, R2, C2, R3, C3),

    % 8. PATH_CLEAR_IGNORING: Caminos despejados (ignorando la casilla origen)
    path_clear_ignoring(ToR, ToC, R2, C2, FromR, FromC),
    path_clear_ignoring(R2, C2, R3, C3, FromR, FromC),


    % 9. ARITMÉTICA IS: Calcular la puntuación del skewer

    piece_value(FrontType, FrontValue),
    piece_value(AttackerType, AttackerValue),
    piece_value(BehindType, BehindValue),

    %    IS: Evalúa la expresión aritmética
    SkewerScore is BehindValue + FrontValue - AttackerValue.




% Check if a move is a capture and return MVV-LVA score
% MVV-LVA = Most Valuable Victim - Least Valuable Attacker
move_creates_capture(FromR, FromC, ToR, ToC, MvvLvaScore) :-

    % 0. Must be a valid move
    move(FromR, FromC, ToR, ToC),

    % 1. Marcamos al atacante
    piece(AttackerType, AttackerColor, FromR, FromC),

    % 2. Marcamos a la victima, si no existe la consulta falla
    piece(VictimType, VictimColor, ToR, ToC),

    % 3. Colores diferentes
    opposite_color(AttackerColor,VictimColor),

    % 4. Valores
    mvv_lva_piece_value(AttackerType,AttackerValue),
    mvv_lva_piece_value(VictimType, VictimValue),

    % 5. Lo calculamos
    MvvLvaScore is (VictimValue*10) - AttackerValue.




% Check if a move is a pawn promotion
move_creates_promotion(FromR, FromC, ToR, ToC) :-

    % 0. Must be a valid move
    move(FromR, FromC, ToR, ToC),

    piece(p,white, FromR , FromC), %Es un peon blanco
    ToR =:=8.                   %Ha llegado a la última fila para las blancas



move_creates_promotion(FromR, FromC, ToR, ToC) :-

    % 0. Must be a valid move
    move(FromR, FromC, ToR, ToC),

    piece(p,black, FromR , FromC), %Es un peon negro
    ToR =:=1.                   %Ha llegado a la última fila para las negras


